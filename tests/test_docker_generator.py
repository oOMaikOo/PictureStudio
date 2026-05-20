"""Tests für core/docker_generator.py"""
from __future__ import annotations
import os, pytest
from core.docker_generator import DockerGenerator


def test_generate_creates_dockerfile(tmp_path):
    gen = DockerGenerator()
    gen.generate(str(tmp_path))
    assert os.path.isfile(tmp_path / "Dockerfile")


def test_generate_creates_compose(tmp_path):
    gen = DockerGenerator()
    gen.generate(str(tmp_path))
    assert os.path.isfile(tmp_path / "docker-compose.yml")


def test_generate_creates_requirements(tmp_path):
    gen = DockerGenerator()
    gen.generate(str(tmp_path))
    assert os.path.isfile(tmp_path / "requirements_monitor.txt")


def test_generate_creates_start_script(tmp_path):
    gen = DockerGenerator()
    gen.generate(str(tmp_path))
    assert os.path.isfile(tmp_path / "run_monitor.sh")


def test_generate_creates_readme(tmp_path):
    gen = DockerGenerator()
    gen.generate(str(tmp_path))
    assert os.path.isfile(tmp_path / "README_deploy.md")


def test_generate_returns_file_list(tmp_path):
    gen = DockerGenerator()
    files = gen.generate(str(tmp_path))
    assert isinstance(files, list)
    assert len(files) == 5
    for f in files:
        assert os.path.isfile(f)


def test_dockerfile_contains_port(tmp_path):
    gen = DockerGenerator()
    gen.generate(str(tmp_path), api_port=9999)
    content = (tmp_path / "Dockerfile").read_text()
    assert "9999" in content


def test_generate_with_model_path(tmp_path):
    gen = DockerGenerator()
    gen.generate(str(tmp_path), model_path="/some/path/my_model.onnx")
    content = (tmp_path / "Dockerfile").read_text()
    assert "my_model.onnx" in content


def test_compose_contains_port(tmp_path):
    gen = DockerGenerator()
    gen.generate(str(tmp_path), api_port=8888)
    content = (tmp_path / "docker-compose.yml").read_text()
    assert "8888" in content
