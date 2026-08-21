# Marker Matching Benchmark — Web App

把 `extract_pairs.py` + `benchmark_matchers.py` 的功能包成一個網頁 App，可上傳
annotator session JSON、調整參數、比較三種 matcher 的**對位精度**與**效能**，並瀏覽
每一組配對的疊圖（含 RANSAC inlier/outlier 區分）。

## 功能

- 上傳 annotator session JSON（含 `ceItems` / `newItems` / `points`）。
- 參數輸入：`lg-features`、`max-features`、`max-dim`、`min-matches`、`ratio`、
  `ransac-thresh`、`ov-size`。
- 引擎 / 裝置切換（自動偵測本機可用的 OpenVINO 裝置）：
  | UI 名稱 | 內部方法 | 說明 |
  |---|---|---|
  | OpenCV SIFT (CPU) | `sift` | 純 OpenCV，CPU |
  | PyTorch LightGlue (no OpenVINO) | `lightglue` | PyTorch DISK+LightGlue |
  | PyTorch + OpenVINO (CPU) | `openvino` | DISK U-Net 走 OpenVINO CPU |
  | PyTorch + OpenVINO (GPU) | `openvino` | 有 GPU 裝置時才顯示 |
  | PyTorch + OpenVINO (NPU) | `openvino` | 有 NPU 裝置時才顯示 |
- 圖形化輸出：
  - **指標表**：誤差平均/中位數 (px)、`<3px` / `<5px` / `<10px` 成功率、ms/影像、
    CPU 秒/%、尖峰記憶體 MB（每個引擎在獨立子程序量測，數字乾淨）。
  - **每組配對疊圖**：可切換引擎/裝置、上一組/下一組瀏覽。
    - 黃色十字＝手動 ce 標記（左圖）
    - 綠線＝RANSAC inlier 配對；紅線＝RANSAC outlier 配對
    - 右圖綠色＝手動 new 標記（ground truth）；紅色＝投影後標記，兩者距離即誤差

## 開發模式執行（用專案 venv）

```powershell
# 安裝 web 相依（重的 ML 套件沿用專案 ../../requirements.txt）
& 'c:/Git/physicalAI/.venv/Scripts/python.exe' -m pip install -r requirements.txt

# 啟動（會自動開瀏覽器到 http://127.0.0.1:8000/）
& 'c:/Git/physicalAI/.venv/Scripts/python.exe' app.py
```

可用環境變數 `APP_HOST` / `APP_PORT` 調整位址與埠。

## 打包成 exe（one-folder）

```powershell
& 'c:/Git/physicalAI/.venv/Scripts/python.exe' -m pip install pyinstaller
& 'c:/Git/physicalAI/.venv/Scripts/python.exe' -m PyInstaller build_exe.spec --noconfirm
```

產出 `dist/MarkerBenchmark/MarkerBenchmark.exe`，雙擊即啟動並開啟瀏覽器。
整個 `dist/MarkerBenchmark/` 資料夾可壓縮後複製到其他電腦。

### 離線 / 無網路電腦的注意事項

第一次執行時，`openvino` 引擎需要：
1. **DISK/LightGlue 權重**（由 torch hub 下載一次後快取）。
2. **DISK U-Net ONNX**（由 `torch.onnx` 匯出到 `../ov_models/disk_unet_<size>.onnx`）。

要做成完全離線的 exe：
1. 先在有網路的機器上執行一次（任一 `openvino` 引擎），讓權重與 ONNX 產生快取。
2. 確認 `annotator/ov_models/disk_unet_1024.onnx` 已存在（`build_exe.spec` 會自動打包）。
3. torch hub 權重預設快取在 `%USERPROFILE%\.cache\torch\hub`；若目標機器無此快取，
   請一併複製，或改用相同 `ov-size` 讓已打包的 ONNX 直接命中（`sift` 引擎不需任何權重）。

> 若只需要 `sift`（OpenCV），則完全不需網路、不需權重與 ONNX。

## 目錄

```
webapp/
  app.py            FastAPI 後端（含 --worker 子程序分派，供打包後使用）
  vis_core.py       比對 + 每-pair 疊圖 + 乾淨資源量測（可獨立 CLI 執行）
  static/index.html 前端（純 HTML/JS）
  requirements.txt  web 相依
  build_exe.spec    PyInstaller one-folder 設定
  runs/             每次執行的輸出（資料、疊圖、指標 JSON）
```
