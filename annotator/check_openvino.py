import openvino as ov

core = ov.Core()
print("OpenVINO", ov.__version__)
for d in core.available_devices:
    try:
        name = core.get_property(d, "FULL_DEVICE_NAME")
    except Exception as e:
        name = f"(no name: {e})"
    print("DEVICE:", d, "->", name)
