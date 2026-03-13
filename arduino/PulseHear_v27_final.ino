// PulseHear v27_final
// FIRE/BABY: standalone OLED alert
// BACKGROUND: BLE send to Flutter for YAMNet analysis
// Keywords: Flutter sends KEYWORD:word, stored in buffer

#define EIDSP_QUANTIZE_FILTERBANK 0
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <ESP_I2S.h>
#include <PulsehearINMP_inferencing.h>
#include "BluetoothSerial.h"

BluetoothSerial SerialBT;

#define OLED_SDA    5
#define OLED_SCL    6
#define I2S_SCK     7
#define I2S_WS      8
#define I2S_SD      9
#define SAMPLE_RATE 16000U
#define I2S_SHIFT   14

Adafruit_SSD1306 display(128, 64, &Wire, -1);
I2SClass I2S;

#define ML_CONF       0.75f
#define ML_CONF_BABY  0.85f
#define RMS_MIN       0.005f
#define RMS_BABY_MAX  0.012f
#define FRAMES        5
#define COOLDOWN_MS   10000

// --- Keyword buffer from Flutter ---
String keyword_buffer[10];
int keyword_count = 0;

static float audio_buf[EI_CLASSIFIER_RAW_SAMPLE_COUNT];
static const uint32_t sample_buffer_size = 2048;
static int32_t rawBuf[sample_buffer_size * 2];
static int16_t monoBuf[sample_buffer_size];

typedef struct {
  int16_t *buffer;
  volatile uint8_t buf_ready;
  volatile uint32_t buf_count;
  uint32_t n_samples;
} inference_t;

static inference_t inference;
static bool record_status = true;
volatile bool is_cooldown = false;
static unsigned long alert_time = 0;

void oled_show(const char* l1, const char* l2 = nullptr) {
  display.clearDisplay();
  display.setTextColor(WHITE);
  int16_t x, y; uint16_t w, h;
  if (l2) {
    display.setTextSize(1);
    display.getTextBounds(l1, 0, 0, &x, &y, &w, &h);
    display.setCursor((128 - w) / 2, 15);
    display.print(l1);
    display.setTextSize(2);
    display.getTextBounds(l2, 0, 0, &x, &y, &w, &h);
    display.setCursor((128 - w) / 2, 35);
    display.print(l2);
  } else {
    display.setTextSize(2);
    display.getTextBounds(l1, 0, 0, &x, &y, &w, &h);
    display.setCursor((128 - w) / 2, 25);
    display.print(l1);
  }
  display.display();
}

void flush_buffer() {
  inference.buf_count = 0;
  inference.buf_ready = 0;
}

float compute_zcr_cv(const float* buf, int len) {
  int seg = SAMPLE_RATE / 10;
  int ns = len / seg;
  if (ns < 3) return 0.5f;
  if (ns > 16) ns = 16;
  float s[16];
  for (int i = 0; i < ns; i++) {
    uint32_t c = 0;
    int off = i * seg;
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

void setup() {
  Serial.begin(115200);
  Wire.begin(OLED_SDA, OLED_SCL);
  if (!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) for (;;);
  oled_show("PulseHear v27");
  delay(2000);
  SerialBT.begin("PulseHear_v27");
  Serial.println("BT ready: PulseHear_v27");
  I2S.setPins(I2S_SCK, I2S_WS, -1, I2S_SD);
  if (!I2S.begin(I2S_MODE_STD, SAMPLE_RATE,
      I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_STEREO)) while (1);
  inference.buffer = (int16_t*)malloc(
      EI_CLASSIFIER_RAW_SAMPLE_COUNT * sizeof(int16_t));
  if (!inference.buffer) while (1);
  inference.buf_count = 0;
  inference.n_samples = EI_CLASSIFIER_RAW_SAMPLE_COUNT;
  inference.buf_ready = 0;
  xTaskCreate(capture_samples, "cap", 1024 * 32,
      (void*)(uintptr_t)(sample_buffer_size * 2 * sizeof(int32_t)), 10, NULL);
  oled_show("Listening...");
}

void loop() {
  // 1. Receive keywords from Flutter via BLE
  if (SerialBT.available()) {
    String cmd = SerialBT.readStringUntil('\n');
    cmd.trim();
    if (cmd.startsWith("KEYWORD:") && keyword_count < 10) {
      keyword_buffer[keyword_count++] = cmd.substring(8);
      SerialBT.println("OK:" + keyword_buffer[keyword_count - 1]);
      Serial.println("Keyword added: " + keyword_buffer[keyword_count - 1]);
    }
  }

  // 2. Cooldown
  if (is_cooldown) {
    if (millis() - alert_time >= COOLDOWN_MS) {
      is_cooldown = false;
      flush_buffer();
      oled_show("Listening...");
    } else { delay(100); return; }
  }

  int fire_votes = 0, baby_votes = 0;
  float zcv_vals[5] = {};
  int counted = 0;
  float last_rms = 0;

  for (int f = 0; f < FRAMES; f++) {
    while (inference.buf_ready == 0) delay(10);
    float rms = 0;
    for (int i = 0; i < EI_CLASSIFIER_RAW_SAMPLE_COUNT; i++) {
      audio_buf[i] = (float)inference.buffer[i] / 32768.0f;
      rms += audio_buf[i] * audio_buf[i];
    }
    rms = sqrtf(rms / EI_CLASSIFIER_RAW_SAMPLE_COUNT);
    flush_buffer();
    if (rms < RMS_MIN) continue;
    last_rms = rms;
    zcv_vals[counted++] = compute_zcr_cv(audio_buf,
        EI_CLASSIFIER_RAW_SAMPLE_COUNT);
    signal_t signal;
    numpy::signal_from_buffer(audio_buf,
        EI_CLASSIFIER_RAW_SAMPLE_COUNT, &signal);
    ei_impulse_result_t result = {0};
    run_classifier(&signal, &result, false);
    float best_val = 0;
    const char* lbl = nullptr;
    for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
      if (result.classification[ix].value > best_val) {
        best_val = result.classification[ix].value;
        lbl = result.classification[ix].label;
      }
    }
    if (best_val >= ML_CONF) {
      if (strcmp(lbl, "fire_alarm") == 0) fire_votes++;
      if (strcmp(lbl, "baby_crying") == 0 &&
          best_val >= ML_CONF_BABY) baby_votes++;
    }
  }

  if (counted == 0) return;

  float avg_zcv = 0;
  for (int i = 0; i < counted; i++) avg_zcv += zcv_vals[i];
  avg_zcv /= counted;
  float std_zcv = 0;
  for (int i = 0; i < counted; i++) {
    float d = zcv_vals[i] - avg_zcv;
    std_zcv += d * d;
  }
  std_zcv = sqrtf(std_zcv / counted);

  // 3. Decision
  String decision = "background";
  if (baby_votes >= 3) decision = "baby_crying";
  else if (fire_votes >= 4 && std_zcv < 0.03f) decision = "fire_alarm";
  else if (fire_votes == 5) decision = "fire_alarm";

  Serial.printf("Decision: %s | fire:%d baby:%d\n",
      decision.c_str(), fire_votes, baby_votes);

  // 4. Act on decision
  if (decision == "fire_alarm") {
    // Standalone - no phone needed
    oled_show("Detected:", "FIRE ALM");
    is_cooldown = true;
    alert_time = millis();
  } else if (decision == "baby_crying") {
    // Standalone - no phone needed
    oled_show("Detected:", "BABY CRY");
    is_cooldown = true;
    alert_time = millis();
  } else {
    // background -> send to phone if connected
    if (SerialBT.hasClient()) {
      // Send JSON with decision + keyword buffer count
      String json = "{\"type\":\"background\",\"kw_count\":";
      json += String(keyword_count) + "}";
      SerialBT.println(json);
      oled_show("Phone...", "Analyzing");
      is_cooldown = true;
      alert_time = millis();
    }
    // if no phone: keep listening silently
  }
}

static void capture_samples(void* arg) {
  const int32_t to_read = (int32_t)(uintptr_t)arg;
  while (record_status) {
    if (is_cooldown) { delay(10); continue; }
    int bytes = I2S.readBytes((char*)rawBuf, to_read);
    if (bytes <= 0) continue;
    int total = bytes / sizeof(int32_t);
    int mc = 0;
    for (int i = 0; i < total; i += 2) {
      int32_t s = rawBuf[i] >> I2S_SHIFT;
      if (s >  32767) s =  32767;
      if (s < -32768) s = -32768;
      monoBuf[mc++] = (int16_t)s;
    }
    for (int i = 0; i < mc; i++) {
      if (inference.buf_ready == 0) {
        inference.buffer[inference.buf_count++] = monoBuf[i];
        if (inference.buf_count >= inference.n_samples)
          inference.buf_ready = 1;
      }
    }
  }
  vTaskDelete(NULL);
}
