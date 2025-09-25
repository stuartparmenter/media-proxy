# Media Proxy — Home Assistant Add-on

Stream videos, GIFs, still images, and YouTube content to tiny LED/LCD displays (e.g., ESPHome devices using DDP) with smart resizing, optional color-range expansion, and packet pacing. This add-on exposes a WebSocket control API and pushes pixel data over UDP using the DDP format.

> The add-on launches `src/run.py` via `ha-addon/run.py`, which reads add-on options and execs the server with `--host/--port` and an optional `--config`.

---

## What you get

- WebSocket control server (default `:8788`) that accepts `start_stream`, `stop_stream`, and `update` commands.  
- Video decoding via PyAV/FFmpeg with auto-rebuilding filter graph (handles rotation, SAR/PAR, format changes).  
- Smart fit modes (`cover`, `pad`, `auto-*`), optional TV→PC range expansion, and automatic black-bar crop.  
- Image path using `imageio` for stills and animated GIFs.  
- UDP DDP sender with optional packet spreading and pacing for smoother delivery.  

---

## Requirements

- Home Assistant OS / Supervisor (add-on system).  
- Network access from HA to your displays (UDP, typically port 4048).  
- Python/FFmpeg dependencies are bundled in the container (`requirements.txt`, `constraints.txt`).  

---

## Install (as a custom add-on repository)

1. In Home Assistant, open **Settings → Add-ons → Add-on Store**.  
2. Click **⋮ → Repositories** and add:  
   `https://github.com/stuartparmenter/media-proxy`  
3. Find **Media Proxy** in the list and click **Install**.  
4. After install, open the add-on:  
   - Optionally enable **Start on boot** and **Watchdog**.  
   - Add configuration (see below).  
   - Click **Start**, then check the **Log** tab to verify it’s running.  

---

## Configuration

By default the add-on runs on `host: 0.0.0.0`, `port: 8788`.  

You can optionally point it to a server configuration file. If you don’t set one, it defaults to `/config/media-proxy.yaml` — but you’ll need to create that file yourself if you want custom defaults.  

### Example config file (`/config/media-proxy.yaml`)

```yaml
hw:
  prefer: auto

video:
  expand_mode: 2        # 0=never, 1=auto(limited->full), 2=force
  fit: pad              # cover | pad
  autocrop:
    enabled: true
    probe_frames: 8
    luma_thresh: 22
    max_bar_ratio: 0.20
    min_bar_px: 2

playback:
  loop: true

youtube:
  60fps: true           # Try 60fps at 720p first, then fall back to resolution-optimized selection

image:
  method: lanczos       # lanczos | bicubic | bilinear | box | nearest | auto
  gamma_correct: false
  color_correction: true
  unsharp:
    amount: 0.0
    radius: 0.6
    threshold: 2

log:
  level: info
  metrics: true
  rate_ms: 5000
  send_ms: false

net:
  win_timer_res: true
  spread_packets: true
  spread_max_fps: 60
  spread_min_ms: 3.0
  spread_max_sleeps: 0

playback_still:
  burst: 3
  spacing_ms: 100
  tail_s: 2.0
  tail_hz: 2
```

---

## YouTube Optimization

Media Proxy intelligently selects YouTube formats based on your display size and hardware acceleration:

**Resolution Matching:** Automatically selects the most appropriate resolution for your display:
- 64×64 displays use 144p → 240p → 360p streams
- 480×480 displays use 480p → 360p → 720p streams
- Reduces bandwidth usage while maintaining visual quality

**60fps Content:** With `youtube.60fps: true` (default), prioritizes smooth motion:
- Attempts 60fps at 720p first, regardless of display size (YouTube only offers 60fps at 720p+)
- Falls back to resolution-matched formats if 60fps unavailable
- Set to `false` to prioritize bandwidth/CPU efficiency over framerate

**Hardware Acceleration:** Codec selection optimized for your acceleration method:
- **VAAPI:** AV1 → VP9 → H.265 → H.264 (modern codec preference)
- **Quick Sync:** H.265 → H.264 → AV1 (optimized for Intel's HEVC support)
- **CUDA:** AV1 → H.265 → H.264 (RTX 30+ series supports AV1 decode)
- **CPU fallback:** H.264 → VP9 → H.265 (efficiency-focused)

Enable `log.level: debug` to see format selection details.

---

## Using it with ESPHome

See: [lvgl-ddp-stream](https://github.com/stuartparmenter/lvgl-ddp-stream) for ESPHome integration examples.  

---

## Ports & networking

- **WebSocket control:** TCP `8788` (configurable).  
- **DDP to devices:** UDP from the add-on to your displays on the port you specify per stream.  

---

## Logs & troubleshooting

- Open the add-on’s **Log** tab to view server output.  
- You’ll see decode path selection, filter-graph rebuilds, and metrics (fps, pps, jitter, drops).  
- Common issues:  
  - **No frames until page reload:** Ensure your client starts streaming after WebSocket connect.  
  - **YouTube not playing:** The add-on must be able to reach YouTube; outbound internet must be allowed.  
  - **Device not receiving frames:** Check UDP reachability and that pixel format matches your device.  

---
