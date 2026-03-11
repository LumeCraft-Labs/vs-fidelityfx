
# vs-fidelityfx

FidelityFX for VapourSynth

## 功能

### EASU - 放大

```python
fidelityfx.EASU(clip, width=?, height=?, fast=0)
```

- `width`: 输出宽度（必须不低于输入宽度）
- `height`: 输出高度（必须不低于输入高度）
- `fast`: 是否启用快速/低质量模式

### RCAS - 锐化

```python
fidelityfx.RCAS(clip, sharpness=0.2)
```

- `sharpness`: 锐化强度，范围 `[0.0, 2.0]`（默认 `0.2` ，注意 `0.0` 表示最大锐化）

### ChromaticAberration - 色差

```python
fidelityfx.CA(clip, intensity=1.0)
```

- `intensity`: 色差强度，范围 `[0.0, 20.0]`（默认 `1.0` ）

### Vignette - 暗角

```python
fidelityfx.VIG(clip, intensity=1.0)
```

- `intensity`: 暗角强度，范围 `[0.0, 2.0]` （默认 `1.0` ）

### Grain - 胶片颗粒

```python
fidelityfx.GRAIN(clip, scale=1.0, amount=0.05, seed=-1)
```

- `scale`: 颗粒尺度，范围 `[0.01, 20.0]` （默认 `1.0` ）
- `amount`: 颗粒强度，范围 `[0.0, 20.0]` （默认 `0.05` ）
- `seed`: 随机种子，整数类型（默认 `-1` ）
  - `-1` 使用帧序号作为种子（每帧颗粒不同，模拟真实胶片）
  - 大于或等于0使用固定种子（所有帧颗粒相同）

### 处理流程示例

```python
import vapoursynth as vs
from vapoursynth import core

clip = video_in # 1080p
clip = core.resize.Bicubic(clip, format=vs.RGBS, matrix_in_s="709")

# FSR1
clip = core.fidelityfx.EASU(clip, width=2560, height=1440) # 对于EASU也建议使用线性空间输入
clip = core.fidelityfx.RCAS(clip, sharpness=0.1)

# 镜头特效
clip = core.fidelityfx.CA(clip, intensity=1.5)
clip = core.fidelityfx.VIG(clip, intensity=0.8)
clip = core.fidelityfx.GRAIN(clip, amount=0.5)

clip.set_output()
```

## 支持的格式

| 输入格式 | 位深 | EASU | RCAS | CA | VIG | GRAIN |
|------|------|------|------|-----|------|-------|
| RGB24 | 8-bit | ✅ | ✅ | ✅ | ✅ | ✅ |
| RGB30 | 10-bit | ✅ | ✅ | ✅ | ✅ | ✅ |
| RGB48 | 16-bit | ✅ | ✅ | ✅ | ✅ | ✅ |
| RGBH | 16-bit 半精度 | ✅ | ✅ | ✅ | ✅ | ✅ |
| RGBS | 32-bit 浮点 | ✅ | ✅ | ✅ | ✅ | ✅ |
| GRAY8 | 8-bit | ✅ | ✅ | ❌ | ✅ | ✅ |
| GRAY16 | 16-bit | ✅ | ✅ | ❌ | ✅ | ✅ |
| GRAYH | 16-bit 半精度 | ✅ | ✅ | ❌ | ✅ | ✅ |
| GRAYS | 32-bit 浮点 | ✅ | ✅ | ❌ | ✅ | ✅ |

不支持 YUV 格式输入。请先使用 `core.resize.Bicubic(clip, format=vs.RGBH, matrix_in_s="709")` 或类似的进行预转换。

## 编译

### Windows 构建（MSYS2 MinGW-w64）

```bash
pacman -S mingw-w64-x86_64-gcc mingw-w64-x86_64-meson mingw-w64-x86_64-ninja

# PowerShell 临时设置：$env:PATH = "C:\msys64\mingw64\bin;$env:PATH"

cd vs-fidelityfx
meson setup build
meson compile -C build
```

## 许可证

MIT

## 其它

- [AMD FidelityFX SDK](https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK)
- [VapourSynth](http://www.vapoursynth.com/)
