# 打包成 EXE 指南（Marker Matching Benchmark）

把整個 web app 用 **PyInstaller one-folder** 模式打包成 Windows 執行檔，
產出一個可直接複製到其他電腦、雙擊即用的資料夾。

- 打包設定檔：[build_exe.spec](build_exe.spec)
- 進入點：[app.py](app.py)（同時是 PyInstaller 的入口）
- 一般使用說明請見 [README.md](README.md)

---

## 1. 前置需求

| 項目 | 說明 |
|---|---|
| Python 環境 | 專案 venv：`c:/Git/physicalAI/.venv`（Python 3.14） |
| 相依套件 | 已裝好 web 相依 + 重的 ML 套件（openvino / torch / kornia / opencv…） |
| 打包工具 | `pyinstaller` |
| 磁碟空間 | 打包後 `dist/` 約數 GB（torch + openvino 原生庫很大） |

安裝相依與打包工具：

```powershell
Set-Location 'c:/Git/physicalAI/annotator/webapp'

# web 相依（重的 ML 套件沿用專案 ../../requirements.txt）
& 'c:/Git/physicalAI/.venv/Scripts/python.exe' -m pip install -r requirements.txt

# 打包工具
& 'c:/Git/physicalAI/.venv/Scripts/python.exe' -m pip install pyinstaller
```

> 若在 Intel 內網，pip 需要走 proxy：加上
> `--proxy http://proxy-png.intel.com:912`。

---

## 2. （建議）先做離線快取

`openvino` / `lightglue` 引擎第一次執行時需要兩個檔案，先在**有網路**的機器上
產生快取，打包後才能完全離線使用：

1. **DISK / LightGlue 權重**：由 torch hub 下載一次，快取在
   `%USERPROFILE%\.cache\torch\hub`。
2. **DISK U-Net ONNX**：由 `torch.onnx` 匯出到 `../ov_models/disk_unet_<size>.onnx`。

做法：先用開發模式跑一次任一 `openvino` 引擎（見 [README.md](README.md)），
跑完後確認下列檔案存在：

```
annotator/ov_models/disk_unet_1024.onnx
```

`build_exe.spec` 偵測到 `annotator/ov_models/` 存在時，會**自動把 ONNX 一起打包**，
執行檔啟動時再複製到 exe 旁邊的可寫入 `ov_models/`。

> 只用 `sift`（OpenCV）引擎的話，**完全不需要**網路、權重或 ONNX。

### 快取根據什麼建立？會影響換參數/換資料嗎？

離線快取是**根據「模型」建立的，跟你的參數和資料庫無關**：

| 快取 | 依據什麼 | 檔名 / 位置 |
|---|---|---|
| torch 權重（DISK / LightGlue） | 只看**模型架構**（DISK `depth`、LightGlue `disk`） | `%USERPROFILE%\.cache\torch\hub` |
| DISK U-Net ONNX | 只看 **`ov-size`**（模型輸入方形邊長） | `ov_models/disk_unet_<size>.onnx` |

因此：

- **換資料庫（不同 JSON / 不同影像）** → 完全不影響，快取是模型層級、不含任何資料，直接用即可。
- **調這些參數不影響、不需重建快取**：`min-matches`、`ratio`、`ransac-thresh`、
  `max-features`、`lg-features`、`max-dim`（都是執行期才套用）。
- **只有調 `ov-size`** 會觸發**新的 ONNX 匯出**（檔名帶 size，如 `disk_unet_1024.onnx`、
  `disk_unet_1280.onnx`）。換 size 第一次跑會多花幾秒匯出；舊 size 的快取不會被覆蓋，
  之後切回去仍直接命中。

> 小結：快取＝模型指紋，不是資料或參數的指紋。打包前只要把**常用的 `ov-size`**
> 對應的 ONNX 先產生好一起打包，日後換資料庫、調其他參數都不必重做。

---

## 3. 執行打包

```powershell
Set-Location 'c:/Git/physicalAI/annotator/webapp'
& 'c:/Git/physicalAI/.venv/Scripts/python.exe' -m PyInstaller build_exe.spec --noconfirm
```

- `--noconfirm`：覆寫舊的 `build/` 與 `dist/`，不再詢問。
- 首次打包較久（要掃描並收集 torch / openvino 的原生庫）。

打包成功後產出：

```
dist/MarkerBenchmark/
  MarkerBenchmark.exe      ← 雙擊啟動
  _internal/               ← 相依庫、static/、（若有）ov_models/
```

---

## 4. 執行與散佈

- **啟動**：雙擊 `MarkerBenchmark.exe`，會保留一個 console 視窗顯示本機網址與 log，
  並自動開啟瀏覽器到 `http://127.0.0.1:8000/`。
- **散佈**：把整個 `dist/MarkerBenchmark/` 資料夾壓縮後複製到目標電腦即可，
  不需在目標機安裝 Python。
- **輸出位置**：每次執行的結果寫在 **exe 同層** 的 `runs/<run_id>/`
  （資料、疊圖、指標 JSON、log），可用 App 內「載入既有結果」分頁重新開啟。

### 權限說明

- **不需要系統管理員權限**：只綁定 `127.0.0.1:8000`（port > 1024）、只寫入 exe 同層資料夾。
- ⚠️ 若把 exe 放在**受保護目錄**（如 `C:\Program Files`），寫入 `runs/` 會失敗。
  → 建議放在**使用者可寫入的位置**（桌面、文件夾、`D:\` 等）。
- 首次執行時 Windows SmartScreen 可能因未簽章跳出提示，點「仍要執行」即可。

### 自訂位址 / 埠

用環境變數覆寫（預設 `127.0.0.1:8000`）：

```powershell
$env:APP_HOST = '127.0.0.1'
$env:APP_PORT = '8080'
.\MarkerBenchmark.exe
```

---

## 5. one-folder 的 --worker 機制

打包後的 exe 會用一個隱藏的 `--worker` 旗標**重新呼叫自己**，讓每個 matcher 引擎
在乾淨的子程序中執行（資源量測不互相干擾）。這是 [app.py](app.py) 的 `main()` 邏輯，
打包時不需額外設定。

---

## 6. 疑難排解

| 症狀 | 可能原因與解法 |
|---|---|
| 啟動閃退、console 一閃而過 | 從 PowerShell 手動執行 `.\MarkerBenchmark.exe` 看錯誤訊息 |
| `ModuleNotFoundError: xxx` | 在 `build_exe.spec` 的 `hiddenimports` 補上該模組後重打包 |
| openvino 引擎失敗（無 ONNX） | 依第 2 節先產生 `ov_models/disk_unet_<size>.onnx` 再打包 |
| 無網路機器 openvino 下載失敗 | 一併複製 `%USERPROFILE%\.cache\torch\hub` 權重快取 |
| `runs/` 無法寫入 | exe 放在受保護目錄，改放使用者可寫入位置 |
| 打包體積太大 | 正常（torch + openvino）；可在 spec 的 `excludes` 移除未用套件 |
| 埠被占用 | 用 `APP_PORT` 換一個埠 |

---

## 7. 目錄對照

```
webapp/
  app.py            FastAPI 後端 + PyInstaller 進入點（含 --worker 子程序分派）
  vis_core.py       比對 + 每-pair 疊圖 + 乾淨資源量測（可獨立 CLI）
  static/index.html 前端（純 HTML/JS）
  requirements.txt  web 相依
  build_exe.spec    PyInstaller one-folder 設定
  BUILD_EXE.md      （本檔）打包指南
  README.md         一般使用說明
  runs/             每次執行的輸出（資料、疊圖、指標 JSON、log）
```
