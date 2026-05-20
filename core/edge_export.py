from __future__ import annotations
import logging
import os
from typing import Optional

log = logging.getLogger(__name__)


class EdgeExporter:
    """Exportiert PyTorch-Klassifikationsmodelle für Edge-Deployment."""

    def export_quantized_onnx(
        self,
        model_path: str,
        output_path: str,
        image_size: int = 224,
        quantize: bool = True,
    ) -> str:
        """
        Exportiert .pth Checkpoint zu ONNX, optional INT8-quantisiert.
        Gibt output_path zurück.
        """
        try:
            import torch
        except ImportError as exc:
            raise ImportError("torch ist nicht installiert.") from exc

        model = self._load_model(model_path)
        model.eval()
        dummy = torch.randn(1, 3, image_size, image_size)

        # Zunächst zu temp ONNX exportieren
        base_path = output_path if not quantize else output_path.replace(".onnx", "_fp32.onnx")
        torch.onnx.export(
            model,
            dummy,
            base_path,
            export_params=True,
            opset_version=14,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        )

        if quantize:
            try:
                from onnxruntime.quantization import quantize_dynamic, QuantType
                quantize_dynamic(base_path, output_path, weight_type=QuantType.QInt8)
                try:
                    os.remove(base_path)
                except OSError:
                    pass
                log.info("INT8-quantisiertes ONNX gespeichert: %s", output_path)
            except ImportError:
                log.warning("onnxruntime.quantization nicht verfügbar – plain ONNX gespeichert.")
                if base_path != output_path:
                    os.rename(base_path, output_path)
        return output_path

    def export_coreml(
        self,
        model_path: str,
        output_path: str,
        image_size: int = 224,
    ) -> str:
        """
        Exportiert zu CoreML (.mlpackage). Nur macOS + coremltools.
        Raises ImportError wenn coremltools nicht installiert.
        """
        try:
            import coremltools as ct
        except ImportError as exc:
            raise ImportError(
                "coremltools ist nicht installiert (pip install coremltools). "
                "Nur auf macOS verfügbar."
            ) from exc
        try:
            import torch
        except ImportError as exc:
            raise ImportError("torch ist nicht installiert.") from exc

        model = self._load_model(model_path)
        model.eval()
        dummy = torch.randn(1, 3, image_size, image_size)
        traced = torch.jit.trace(model, dummy)
        cml = ct.convert(
            traced,
            inputs=[ct.TensorType(name="input", shape=dummy.shape)],
            convert_to="mlprogram",
        )
        cml.save(output_path)
        log.info("CoreML gespeichert: %s", output_path)
        return output_path

    def _load_model(self, model_path: str):
        """Lädt PyTorch-Modell aus Checkpoint (.pth)."""
        import torch
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

        if isinstance(checkpoint, dict):
            # Standard PictureStudio checkpoint
            from core.model_manager import ModelManager
            mm = ModelManager()
            arch = checkpoint.get("model_type", "resnet18")
            n_classes = checkpoint.get("num_classes", 2)
            model = mm._build_model(arch, n_classes, pretrained=False)
            state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
            model.load_state_dict(state)
        else:
            model = checkpoint

        return model

    @staticmethod
    def has_coreml() -> bool:
        """True wenn coremltools importierbar."""
        try:
            import coremltools
            return True
        except ImportError:
            return False

    @staticmethod
    def has_quantization() -> bool:
        """True wenn onnxruntime.quantization verfügbar."""
        try:
            from onnxruntime.quantization import quantize_dynamic
            return True
        except ImportError:
            return False
