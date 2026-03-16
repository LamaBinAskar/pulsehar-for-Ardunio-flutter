/* ============================================================
   PulseHear v34
   ── What's new vs v33 ───────────────────────────────────────
     • RULE ENFORCED: Fire alarm and baby cry are classified
       ONLY by the ESP32 Edge Impulse model — NEVER streamed
       to phone for YAMNet.
     • YAMNet path now explicitly skips streaming if ANY
       fire or baby votes occurred in the cycle (even if below
       threshold). Only truly clean background cycles are sent.
     • Phone YAMNet only classifies: siren, ambulance, police,
       dog bark, doorbell, knock, phone ring, music, speech,
       scream, shout, laughter.
     • No detection threshold changes from v25c_final_tuned.
   ── Detection base ──────────────────────────────────────────
     v25c_final_tuned — DO NOT CHANGE thresholds
   ── OLED ────────────────────────────────────────────────────
     Boot          : "Listening"
     Fire          : "Detected:" / "FIRE ALM"
     Baby          : "Detected:" / "BABY CRY"
     Mixed         : "Warning:"  / "MIXED ALM"
     YAMNet result : "YAMNet:"   / <label>
     Keyword       : "Keyword:"  / <word>
   ── Detection split ─────────────────────────────────────────
     ESP32 (Edge Impulse) → fire_alarm, baby_crying, mixed
       • Sends BLE signal: "FIRE" | "BABY" | "MIXED"
       • Works fully offline — phone not required
     Phone (YAMNet via BLE audio stream) → background sounds
       • Only receives audio when cycle = pure background
         (zero fire votes AND zero baby votes)
       • Classifies: siren, doorbell, dog bark, phone, etc.
   ── BLE ─────────────────────────────────────────────────────
     SIGNAL  (notify) : ESP32 → phone  "FIRE" | "BABY" | "MIXED"
     KEYWORD (write)  : phone → ESP32  "RESULT:xxx" | "KW:word"
     AUDIO   (notify) : ESP32 → phone  raw PCM (background only)
   ── UUIDs — must match Flutter bluetooth_service.dart ───────
     Service  : 12345678-1234-1234-1234-123456789abc
     Signal   : abcd1234-1234-1234-1234-abcdef123456
     Keyword  : abcd5678-1234-1234-1234-abcdef123456
     Audio    : abcd9999-1234-1234-1234-abcdef123456
   ============================================================ */

#define EIDSP_QUANTIZE_FILTERBANK 0

#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <ESP_I2S.h>
#include <PulsehearINMP_inferencing.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>

// ── Pin map ──────────────────────────────────────────────────
#define OLED_SDA    5
#define OLED_SCL    6
#define I2S_SCK     7
#define I2S_WS      8
#define I2S_SD      9
#define SAMPLE_RATE 16000U
#define I2S_SHIFT   14

// ── Detection thresholds (v25c_final_tuned — DO NOT CHANGE) ──
#define ML_CONF           0.75f
#define ML_CONF_BABY      0.85f
#define ZCV_AVG_MAX       0.20f   // Baby cry avg_zcv ~ 0.30, fire ~ 0.08
#define ZCV_STD_MAX       0.03f   // Baby cry is irregular, fire is steady
#define RMS_MIN           0.005f  // Silence gate
#define RMS_BABY_MAX      0.12f   // Baby cry RMS upper limit
#define FRAMES            5       // Frames per detection cycle
#define FIRE_VOTES_NEEDED 4       // Need 4/5 frames to confirm fire
#define BABY_VOTES_NEEDED 3       // Need 3/5 frames to confirm baby
#define COOLDOWN_MS       10000   // 10 s pause after alert

// ── Background → YAMNet settings ─────────────────────────────
// Audio is streamed to phone ONLY when:
//   1. decision == background (no fire/baby/mixed)
//   2. fire_votes == 0 AND baby_votes == 0  (pure background cycle)
//   3. background frame confidence >= BG_CONF_MIN
#define BG_CONF_MIN         0.65f   // Min background confidence to save frame
#define RESULT_TIMEOUT_MS   8000    // Max wait for phone YAMNet reply (ms)
#define YAMNET_THROTTLE_MS  12000   // Min gap between YAMNet calls (ms)

// ── BLE identifiers (must match Flutter exactly) ─────────────
#define DEVICE_NAME       "PulseHear_v30"
#define SERVICE_UUID      "12345678-1234-1234-1234-123456789abc"
#define CHAR_SIGNAL_UUID  "abcd1234-1234-1234-1234-abcdef123456"
#define CHAR_KEYWORD_UUID "abcd5678-1234-1234-1234-abcdef123456"
#define CHAR_AUDIO_UUID   "abcd9999-1234-1234-1234-abcdef123456"

// ── Hardware ─────────────────────────────────────────────────
Adafruit_SSD1306 display(128, 64, &Wire, -1);
I2SClass I2S;

// ── BLE handles ──────────────────────────────────────────────
BLEServer*         bleServer       = nullptr;
BLECharacteristic* signalChar      = nullptr;
BLECharacteristic* keywordChar     = nullptr;
BLECharacteristic* audioChar       = nullptr;
bool               deviceConnected = false;

// ── YAMNet result (set by BLE callback, read by loop) ────────
volatile bool waitingForResult = false;
String        yamnetResult     = "";

// ── Audio buffers ─────────────────────────────────────────────
static float         audio_buf[EI_CLASSIFIER_RAW_SAMPLE_COUNT];
static const int32_t sample_buffer_size = 2048;
static int32_t       rawBuf[sample_buffer_size * 2];
static int16_t       monoBuf[sample_buffer_size];
static int16_t       bleSendBuf[SAMPLE_RATE];   // best background frame
static float         bestBgConf  = 0;
static int           savedBgBytes = 0;

// ── YAMNet throttle ───────────────────────────────────────────
static unsigned long lastYamnetSend = 0;

// ── Inference buffer (filled by capture task) ─────────────────
typedef struct {
  int16_t*          buffer;
  volatile uint8_t  buf_ready;
  volatile uint32_t buf_count;
  uint32_t          n_samples;
} inference_t;

static inference_t   inference;
static bool          record_status = true;
volatile bool        is_cooldown   = false;
static unsigned long alert_time    = 0;

// ── Forward declarations ──────────────────────────────────────
void oled_show(const char* line1, const char* line2 = nullptr);
void oled_yamnet(const char* result);
static void capture_samples(void* arg);

// ════════════════════════════════════════════════════════════
//  BLE CALLBACKS
// ════════════════════════════════════════════════════════════
class ServerCallbacks : public BLEServerCallbacks {
  void onConnect(BLEServer*) override {
    deviceConnected = true;
    Serial.println("[BLE] Phone connected");
  }
  void onDisconnect(BLEServer*) override {
    deviceConnected = false;
    Serial.println("[BLE] Phone disconnected — re-advertising");
    BLEDevice::startAdvertising();
  }
};

class KeywordCallbacks : public BLECharacteristicCallbacks {
  void onWrite(BLECharacteristic* pChar) override {
    String value = String(pChar->getValue().c_str());
    value.trim();
    Serial.print("[BLE] From phone: "); Serial.println(value);

    if (value.startsWith("RESULT:")) {
      // YAMNet background classification result from phone
      yamnetResult = value.substring(7);
      yamnetResult.trim();
      waitingForResult = false;

    } else if (value.startsWith("KW:")) {
      // Vosk keyword detected on phone — show on OLED
      String kw = value.substring(3);
      kw.trim();
      Serial.println("[KW] Keyword detected: " + kw);
      oled_show("Keyword:", kw.c_str());
      delay(3000);
      oled_show("Listening");
    }
  }
};

// ════════════════════════════════════════════════════════════
//  OLED HELPERS
// ════════════════════════════════════════════════════════════
void oled_show(const char* line1, const char* line2) {
  display.clearDisplay();
  display.setTextColor(WHITE);
  int16_t x, y; uint16_t w, h;
  if (line2) {
    display.setTextSize(1);
    display.getTextBounds(line1, 0, 0, &x, &y, &w, &h);
    display.setCursor((128 - w) / 2, 15); display.print(line1);
    display.setTextSize(2);
    display.getTextBounds(line2, 0, 0, &x, &y, &w, &h);
    display.setCursor((128 - w) / 2, 35); display.print(line2);
  } else {
    display.setTextSize(2);
    display.getTextBounds(line1, 0, 0, &x, &y, &w, &h);
    display.setCursor((128 - w) / 2, 25); display.print(line1);
  }
  display.display();
}

// Split YAMNet result at space so both words fit at text size 2
void oled_yamnet(const char* result) {
  display.clearDisplay();
  display.setTextColor(WHITE);
  String s = String(result);
  int sp = s.indexOf(' ');
  int16_t x, y; uint16_t w, h;
  if (sp > 0) {
    String w1 = s.substring(0, sp);
    String w2 = s.substring(sp + 1);
    display.setTextSize(1);
    display.getTextBounds("YAMNet:", 0, 0, &x, &y, &w, &h);
    display.setCursor((128 - w) / 2, 0); display.print("YAMNet:");
    display.setTextSize(2);
    display.getTextBounds(w1.c_str(), 0, 0, &x, &y, &w, &h);
    display.setCursor((128 - w) / 2, 16); display.print(w1);
    display.getTextBounds(w2.c_str(), 0, 0, &x, &y, &w, &h);
    display.setCursor((128 - w) / 2, 40); display.print(w2);
  } else {
    display.setTextSize(1);
    display.getTextBounds("YAMNet:", 0, 0, &x, &y, &w, &h);
    display.setCursor((128 - w) / 2, 10); display.print("YAMNet:");
    display.setTextSize(2);
    display.getTextBounds(result, 0, 0, &x, &y, &w, &h);
    display.setCursor((128 - w) / 2, 32); display.print(result);
  }
  display.display();
}

// ════════════════════════════════════════════════════════════
//  BLE HELPERS
// ════════════════════════════════════════════════════════════
void sendBLE(const char* signal) {
  if (!deviceConnected) {
    Serial.printf("[BLE] Not connected — skip: %s\n", signal);
    return;
  }
  signalChar->setValue((uint8_t*)signal, strlen(signal));
  signalChar->notify();
  Serial.printf("[BLE] → %s\n", signal);
}

void sendAudioViaBLE(int byteCount) {
  if (!deviceConnected || byteCount < 1) return;
  uint8_t* ptr  = (uint8_t*)bleSendBuf;
  const int CHUNK = 500;

  char startMsg[32];
  snprintf(startMsg, sizeof(startMsg), "AUDIO_START:%d", byteCount);
  audioChar->setValue((uint8_t*)startMsg, strlen(startMsg));
  audioChar->notify();
  delay(100);

  int offset = 0;
  while (offset < byteCount) {
    int toSend = min(CHUNK, byteCount - offset);
    audioChar->setValue(ptr + offset, toSend);
    audioChar->notify();
    offset += toSend;
    delay(15);
  }

  const char* endMsg = "AUDIO_END";
  audioChar->setValue((uint8_t*)endMsg, strlen(endMsg));
  audioChar->notify();
  Serial.printf("[BLE] Audio sent: %d bytes\n", byteCount);
}

void setup_ble() {
  BLEDevice::init(DEVICE_NAME);
  BLEDevice::setMTU(512);
  bleServer = BLEDevice::createServer();
  bleServer->setCallbacks(new ServerCallbacks());

  BLEService* svc = bleServer->createService(SERVICE_UUID);

  signalChar = svc->createCharacteristic(
      CHAR_SIGNAL_UUID, BLECharacteristic::PROPERTY_NOTIFY);
  signalChar->addDescriptor(new BLE2902());

  keywordChar = svc->createCharacteristic(
      CHAR_KEYWORD_UUID,
      BLECharacteristic::PROPERTY_WRITE | BLECharacteristic::PROPERTY_WRITE_NR);
  keywordChar->setCallbacks(new KeywordCallbacks());

  audioChar = svc->createCharacteristic(
      CHAR_AUDIO_UUID, BLECharacteristic::PROPERTY_NOTIFY);
  audioChar->addDescriptor(new BLE2902());

  svc->start();
  BLEAdvertising* adv = BLEDevice::getAdvertising();
  adv->addServiceUUID(SERVICE_UUID);
  adv->setScanResponse(true);
  adv->setMinPreferred(0x06);
  BLEDevice::startAdvertising();
  Serial.println("[BLE] Advertising as: " DEVICE_NAME);
}

// ════════════════════════════════════════════════════════════
//  AUDIO CAPTURE
// ════════════════════════════════════════════════════════════
void flush_buffer() {
  inference.buf_count = 0;
  inference.buf_ready = 0;
}

float compute_zcr_cv(const float* buf, int len) {
  int seg = SAMPLE_RATE / 10;
  int ns  = len / seg;
  if (ns < 3)  return 0.5f;
  if (ns > 16) ns = 16;
  float s[16];
  for (int i = 0; i < ns; i++) {
    uint32_t c = 0; int off = i * seg;
    for (int j = 1; j < seg; j++)
      if ((buf[off+j] >= 0 && buf[off+j-1] < 0) ||
          (buf[off+j] <  0 && buf[off+j-1] >= 0)) c++;
    s[i] = (float)c;
  }
  float mean = 0;
  for (int i = 0; i < ns; i++) mean += s[i];
  mean /= ns;
  if (mean < 1.0f) return 0.5f;
  float var = 0;
  for (int i = 0; i < ns; i++) { float d = s[i] - mean; var += d * d; }
  return sqrtf(var / ns) / mean;
}

// FreeRTOS capture task — pinned to Core 1 (BLE runs on Core 0)
static void capture_samples(void* arg) {
  const int32_t to_read = (int32_t)(uintptr_t)arg;
  while (record_status) {
    if (is_cooldown) { delay(10); continue; }
    int bytes = I2S.readBytes((char*)rawBuf, to_read);
    if (bytes <= 0) continue;
    int total = bytes / sizeof(int32_t), mc = 0;
    for (int i = 0; i < total; i += 2) {
      int32_t s = rawBuf[i] >> I2S_SHIFT;
      if (s >  32767) s =  32767;
      if (s < -32768) s = -32768;
      monoBuf[mc++] = (int16_t)s;
    }
    if (!is_cooldown) {
      for (int i = 0; i < mc; i++) {
        if (inference.buf_ready == 0) {
          inference.buffer[inference.buf_count++] = monoBuf[i];
          if (inference.buf_count >= inference.n_samples)
            inference.buf_ready = 1;
        }
      }
    }
  }
  vTaskDelete(NULL);
}

// ════════════════════════════════════════════════════════════
//  SETUP
// ════════════════════════════════════════════════════════════
void setup() {
  Serial.begin(115200);
  Wire.begin(OLED_SDA, OLED_SCL);
  if (!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) { for (;;); }

  // ── I2S / Microphone FIRST — must start before BLE stack ──
  // (BLE init blocks audio DMA if started first)
  I2S.setPins(I2S_SCK, I2S_WS, -1, I2S_SD);
  if (!I2S.begin(I2S_MODE_STD, SAMPLE_RATE,
                 I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_STEREO)) {
    Serial.println("[ERROR] I2S init failed"); while (1);
  }
  Serial.println("[I2S] Mic ready");

  // ── Inference buffer ───────────────────────────────────────
  inference.buffer = (int16_t*)malloc(
      EI_CLASSIFIER_RAW_SAMPLE_COUNT * sizeof(int16_t));
  if (!inference.buffer) { Serial.println("[ERROR] malloc failed"); while (1); }
  inference.buf_count = 0;
  inference.n_samples = EI_CLASSIFIER_RAW_SAMPLE_COUNT;
  inference.buf_ready = 0;
  record_status       = true;
  ei_sleep(100);

  // ── Print model info ───────────────────────────────────────
  Serial.println("[MODEL] Labels:");
  for (int i = 0; i < (int)EI_CLASSIFIER_LABEL_COUNT; i++)
    Serial.printf("  [%d] %s\n", i, ei_classifier_inferencing_categories[i]);

  // ── Capture task (Core 1 — BLE stays on Core 0) ───────────
  xTaskCreatePinnedToCore(capture_samples, "cap", 1024 * 32,
              (void*)(uintptr_t)(sample_buffer_size * 2 * sizeof(int32_t)),
              10, NULL, 1);
  Serial.println("[CAPTURE] Task started on Core 1");

  // ── BLE — started AFTER mic & capture task ─────────────────
  setup_ble();

  // ── OLED boot message ──────────────────────────────────────
  oled_show("Listening");
  Serial.println("\n=== PulseHear v34 ready ===");
  Serial.printf("  FIRE_VOTES_NEEDED : %d\n", FIRE_VOTES_NEEDED);
  Serial.printf("  BABY_VOTES_NEEDED : %d\n", BABY_VOTES_NEEDED);
  Serial.printf("  COOLDOWN          : %d ms\n", COOLDOWN_MS);
  Serial.printf("  YAMNET_THROTTLE   : %d ms\n", YAMNET_THROTTLE_MS);
  Serial.printf("  BG_CONF_MIN       : %.2f\n\n", BG_CONF_MIN);
}

// ════════════════════════════════════════════════════════════
//  MAIN LOOP
// ════════════════════════════════════════════════════════════
void loop() {

  // ── Cooldown after fire / baby alert ─────────────────────
  if (is_cooldown) {
    if (millis() - alert_time >= COOLDOWN_MS) {
      is_cooldown = false;
      flush_buffer();
      oled_show("Listening");
      Serial.println("--- Resumed ---\n");
    } else {
      delay(100);
      return;
    }
  }

  // ── Per-cycle accumulators ────────────────────────────────
  int   fire_votes = 0, baby_votes = 0;
  float zcv_vals[FRAMES] = {};
  int   counted          = 0;
  float last_rms         = 0;

  // Background tracking for YAMNet path
  float bestBgConf_local = 0;
  bestBgConf   = 0;
  savedBgBytes = 0;

  Serial.println("--- New Cycle ---");

  // ── 5-frame detection loop (v25c_final_tuned — unchanged) ─
  for (int f = 0; f < FRAMES; f++) {

    while (inference.buf_ready == 0) delay(10);

    float rms = 0;
    for (int i = 0; i < EI_CLASSIFIER_RAW_SAMPLE_COUNT; i++) {
      audio_buf[i] = (float)inference.buffer[i] / 32768.0f;
      rms += audio_buf[i] * audio_buf[i];
    }
    rms = sqrtf(rms / EI_CLASSIFIER_RAW_SAMPLE_COUNT);
    flush_buffer();

    if (rms < RMS_MIN) {
      Serial.printf("  Frame %d: Very quiet (RMS:%.4f)\n", f + 1, rms);
      continue;
    }
    last_rms = rms;

    float zcv = compute_zcr_cv(audio_buf, EI_CLASSIFIER_RAW_SAMPLE_COUNT);
    zcv_vals[counted] = zcv;
    counted++;

    signal_t signal;
    numpy::signal_from_buffer(audio_buf, EI_CLASSIFIER_RAW_SAMPLE_COUNT, &signal);
    ei_impulse_result_t result = { 0 };
    run_classifier(&signal, &result, false);

    // Find top class
    float best_val = 0;
    const char* lbl = nullptr;
    float bg_val    = 0;
    for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
      float v = result.classification[ix].value;
      if (v > best_val) { best_val = v; lbl = result.classification[ix].label; }
      if (strcmp(result.classification[ix].label, "background") == 0) bg_val = v;
    }

    // Debug output
    Serial.printf("  Frame %d: ", f + 1);
    for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++)
      Serial.printf("%s:%.0f%% ", result.classification[ix].label,
                    result.classification[ix].value * 100);
    Serial.printf("| ZCV:%.2f RMS:%.4f\n", zcv, rms);

    // Vote counting (v25c_final_tuned — unchanged)
    if (best_val >= ML_CONF) {
      if (strcmp(lbl, "fire_alarm")  == 0) fire_votes++;
      if (strcmp(lbl, "baby_crying") == 0 && best_val >= ML_CONF_BABY)
        baby_votes++;
    }

    // Save highest-confidence background frame for YAMNet.
    // This frame is only ever sent to phone if fire_votes == 0
    // AND baby_votes == 0 at end of cycle (see streaming guard below).
    if (bg_val > bestBgConf_local) {
      bestBgConf_local = bg_val;
      if (bg_val >= BG_CONF_MIN) {
        bestBgConf = bg_val;
        int n = min((int)EI_CLASSIFIER_RAW_SAMPLE_COUNT, (int)SAMPLE_RATE);
        for (int i = 0; i < n; i++)
          bleSendBuf[i] = (int16_t)(audio_buf[i] * 32768.0f);
        savedBgBytes = n * sizeof(int16_t);
      }
    }
  }

  if (counted == 0) {
    Serial.println("  All quiet\n");
    return;
  }

  // ── ZCR statistics ────────────────────────────────────────
  float avg_zcv = 0;
  for (int i = 0; i < counted; i++) avg_zcv += zcv_vals[i];
  avg_zcv /= counted;

  float std_zcv = 0;
  for (int i = 0; i < counted; i++) {
    float d = zcv_vals[i] - avg_zcv;
    std_zcv += d * d;
  }
  std_zcv = sqrtf(std_zcv / counted);

  Serial.printf("  Votes→ Fire:%d Baby:%d | ZCV avg:%.2f std:%.3f | RMS:%.4f\n",
                fire_votes, baby_votes, avg_zcv, std_zcv, last_rms);

  // ── Decision (v25c_final_tuned — unchanged logic) ─────────
  String decision = "", reason = "";

  // 1. Baby crying — direct ML vote
  if (baby_votes >= BABY_VOTES_NEEDED) {
    decision = "baby";
    reason   = "ML baby (" + String(baby_votes) + "/5 frames)";
  }
  // 2. ZCR baby fallback:
  //    Model outputs fire_alarm but ZCR pattern matches baby cry
  //    Baby cry: avg_zcv ~0.30 (high), std_zcv > 0.03 (erratic)
  //    Fire alarm: avg_zcv ~0.08 (low), std_zcv < 0.03 (steady beep)
  else if (fire_votes >= BABY_VOTES_NEEDED && baby_votes == 0
           && std_zcv > ZCV_STD_MAX && avg_zcv > ZCV_AVG_MAX
           && last_rms <= RMS_BABY_MAX) {
    decision = "baby";
    reason   = "Fire+high avgZCV(" + String(avg_zcv, 2) +
               ")+high stdZCV(" + String(std_zcv, 3) + ")=baby";
  }
  // 3. Fire alarm — stable beep pattern (consistent ZCR)
  else if (fire_votes >= FIRE_VOTES_NEEDED && std_zcv < ZCV_STD_MAX) {
    decision = "fire";
    reason   = "ML fire " + String(fire_votes) + "/5 + stable ZCV";
  }
  // 4. Fire alarm — unanimous (all 5 frames)
  else if (fire_votes == FRAMES && fire_votes > baby_votes) {
    decision = "fire";
    reason   = "ML unanimous fire (5/5)";
  }
  // 5. Mixed — some fire + some baby votes
  else if (fire_votes >= 2 && baby_votes >= 1) {
    decision = "mixed";
    reason   = "ML mixed fire/baby";
  }

  // ════════════════════════════════════════════════════════
  //  ACT ON DECISION
  // ════════════════════════════════════════════════════════

  if (decision == "fire") {
    // ── Fire alarm: ESP32 handles entirely — no audio streamed ──
    Serial.printf(">>> FIRE ALARM [%s]\n\n", reason.c_str());
    oled_show("Detected:", "FIRE ALM");
    sendBLE("FIRE");
    is_cooldown = true; alert_time = millis();

  } else if (decision == "baby") {
    // ── Baby cry: ESP32 handles entirely — no audio streamed ────
    Serial.printf(">>> BABY CRYING [%s]\n\n", reason.c_str());
    oled_show("Detected:", "BABY CRY");
    sendBLE("BABY");
    is_cooldown = true; alert_time = millis();

  } else if (decision == "mixed") {
    // ── Mixed: ESP32 handles entirely — no audio streamed ───────
    Serial.printf(">>> MIXED ALARM [%s]\n\n", reason.c_str());
    oled_show("Warning:", "MIXED ALM");
    sendBLE("MIXED");
    is_cooldown = true; alert_time = millis();

  } else {
    // ── Background: stream audio to phone for YAMNet ─────────────
    // RULE: only send if this was a PURE background cycle.
    // Any fire or baby vote — even below threshold — means the
    // sound is ambiguous and must NOT go to phone YAMNet.
    Serial.println(">>> Background cycle");

    if (fire_votes > 0 || baby_votes > 0) {
      // Ambiguous cycle — fire/baby votes present but below threshold.
      // ESP32 didn't confirm, but it's not clean background either.
      // Do not stream to YAMNet.
      Serial.printf("[BG] Ambiguous cycle (fire:%d baby:%d votes) — skip YAMNet\n",
                    fire_votes, baby_votes);

    } else if (!deviceConnected) {
      Serial.println("[BG] No BLE — continuing to listen");

    } else if (millis() - lastYamnetSend < YAMNET_THROTTLE_MS) {
      Serial.printf("[BG] Throttled (%lu ms remaining)\n",
                    YAMNET_THROTTLE_MS - (millis() - lastYamnetSend));

    } else if (savedBgBytes >= 100) {
      // Pure background cycle with good confidence — send to YAMNet
      Serial.printf("[BG] Sending audio to phone (bg conf:%.2f)...\n", bestBgConf);
      lastYamnetSend   = millis();
      waitingForResult = true;
      yamnetResult     = "";

      sendAudioViaBLE(savedBgBytes);

      unsigned long waitStart = millis();
      while (waitingForResult && millis() - waitStart < RESULT_TIMEOUT_MS) {
        delay(50);
      }

      if (yamnetResult.length() > 0) {
        Serial.println("[RESULT] YAMNet → " + yamnetResult);
        oled_yamnet(yamnetResult.c_str());
        delay(3000);
        oled_show("Listening");
      } else {
        Serial.println("[RESULT] Timeout — no reply from phone");
      }
      flush_buffer();

    } else {
      Serial.println("[BG] No background audio saved this cycle\n");
    }
  }
}

// ── Sanity check: model must be trained for microphone ───────
#if !defined(EI_CLASSIFIER_SENSOR) || \
    EI_CLASSIFIER_SENSOR != EI_CLASSIFIER_SENSOR_MICROPHONE
#error "Loaded model is not for a microphone sensor."
#endif
