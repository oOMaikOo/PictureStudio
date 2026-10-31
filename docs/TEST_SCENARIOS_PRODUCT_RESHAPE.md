# Test scenarios — Product reshape

Manual + automated coverage for the Studio/Monitor reshape.

## Automated (`tests/test_product_reshape.py`)

| ID | Scenario | Expected |
|----|----------|----------|
| U1a | Version parse / compare | `v2.6.0` > `2.5.2` |
| U1b | CHANGELOG What's new | Unreleased body extracted; older sections skipped |
| U1c | GitHub tag fetch (mocked) | Returns `tag_name` |
| U1d | Update banner UI | Visible with remote version text |
| U2 | Review hub lists | Uncertain + AL + batch low-conf rows |
| D1a | Product line settings | studio/monitor persist |
| D1b | Sidebar Studio vs Monitor | Studio: labeling; Monitor: camera/anomaly train; no cross-leak |
| D2 | Advanced fold | Edge panel / SSH / HPT hidden until enabled |
| D3 | Live → Train | Camera `_goto_anomaly_training` → `Page.ANOMALY_TRAINING` |
| U3a | Camera list smoke | Mocked `list_usb_cameras` returns fake device |
| U3b | Windows packaging files | `build_windows.bat` + Inno `.iss` exist |
| INT | MainWindow | Review page + update banner present |

## Manual smoke (Windows VM)

1. Run `build_windows.bat` → `dist\PictureStudio\PictureStudio.exe` starts.
2. First launch: Product dialog (Studio / Monitor) then wizard.
3. Studio: Review-Queue visible; Data Drift only under Ansicht → Erweitert.
4. Monitor: Anomalie-Training + Live-Scoring; no Labeling in sidebar.
5. Live-Scoring → `→ Anomalie-Training` opens train page.
6. Optional: install via Inno Setup once `iscc` is available.
