"""Pytest tests for segment_then_name.py.

Run from the directory containing segment_then_name.py and
segment_and_name_sam3.py:

    python -m pytest -v

The tests mock model pipelines and model loading; they do not download models
or require a GPU. The application dependencies (torch, transformers, Pillow,
NumPy) and the base module must still be importable.
"""
from types import SimpleNamespace
import json

import numpy as np
import pytest
from PIL import Image

import segment_then_name as sut


@pytest.fixture
def rgb_image():
    """Small deterministic RGB image."""
    return np.full((20, 30, 3), 200, dtype=np.uint8)


@pytest.fixture
def sample_mask():
    mask = np.zeros((20, 30), dtype=bool)
    mask[5:10, 8:14] = True
    return mask


def make_instance(mask, name=None, score=0.9):
    return sut.Instance(name, mask.copy(), score)


# ---------------------------------------------------------------------------
# Pure utility functions
# ---------------------------------------------------------------------------

def test_mask_iou_identical_masks(sample_mask):
    assert sut.mask_iou(sample_mask, sample_mask) == pytest.approx(1.0)


def test_mask_iou_disjoint_masks():
    a = np.zeros((5, 5), dtype=bool)
    b = np.zeros((5, 5), dtype=bool)
    a[:2, :2] = True
    b[3:, 3:] = True
    assert sut.mask_iou(a, b) == 0.0


def test_mask_iou_partial_overlap():
    a = np.zeros((4, 4), dtype=bool)
    b = np.zeros((4, 4), dtype=bool)
    a[:2, :2] = True
    b[1:3, 1:3] = True
    # Intersection=1, union=7.
    assert sut.mask_iou(a, b) == pytest.approx(1 / 7)


def test_mask_iou_empty_masks():
    empty = np.zeros((3, 3), dtype=bool)
    assert sut.mask_iou(empty, empty) == 0.0


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("A red chair.", "red chair"),
        ("The apple", "apple"),
        ("this is a small lamp", "small lamp"),
        ("  PHOTO OF a wooden table!  ", "wooden table"),
        ("unclear", ""),
        ("a background", ""),
        ("", ""),
        ("x", ""),
        ("a very long description of an object", ""),
        ("A picture of a blue vase\nExtra explanation", "blue vase"),
        ("something unclear nearby", ""),
    ],
)
def test_clean_name(raw, expected):
    assert sut.clean_name(raw) == expected


def test_clean_name_rejects_generic_names():
    for value in ("unknown", "object", "thing", "photo", "texture", "n/a"):
        assert sut.clean_name(value) == ""


def test_softmax_rows_sum_to_one():
    values = np.array([[1.0, 2.0, 3.0], [1000.0, 1001.0, 1002.0]])
    result = sut.softmax(values)
    assert result.shape == values.shape
    np.testing.assert_allclose(result.sum(axis=1), [1.0, 1.0])
    assert np.isfinite(result).all()


def test_softmax_preserves_order():
    result = sut.softmax(np.array([1.0, 2.0, 3.0]))
    assert np.all(np.diff(result) > 0)


# ---------------------------------------------------------------------------
# Image processing
# ---------------------------------------------------------------------------

def test_crop_segment_returns_square_rgb_image(rgb_image, sample_mask):
    crop = sut.crop_segment(rgb_image, sample_mask)
    assert isinstance(crop, Image.Image)
    assert crop.mode == "RGB"
    assert crop.width == crop.height
    assert crop.width >= 6


def test_crop_segment_grays_pixels_outside_mask(rgb_image):
    mask = np.zeros((20, 30), dtype=bool)
    mask[5:10, 10:15] = True
    mask[7, 12] = False
    crop = np.asarray(sut.crop_segment(rgb_image, mask))
    # The object pixels retain the original value; the surrounding crop is gray.
    assert any(np.all(crop == 128, axis=2).ravel())
    assert any(np.all(crop == 200, axis=2).ravel())


def test_crop_segment_handles_mask_at_image_edge(rgb_image):
    mask = np.zeros((20, 30), dtype=bool)
    mask[:4, :5] = True
    crop = sut.crop_segment(rgb_image, mask)
    assert crop.size[0] == crop.size[1]


def test_highlight_segment_changes_only_masked_pixels(rgb_image, sample_mask):
    result = np.asarray(sut.highlight_segment(rgb_image, sample_mask))
    assert result.shape == rgb_image.shape
    np.testing.assert_array_equal(result[~sample_mask], rgb_image[~sample_mask])
    assert not np.array_equal(result[sample_mask], rgb_image[sample_mask])
    assert result[sample_mask, 0].mean() > result[sample_mask, 2].mean()


# ---------------------------------------------------------------------------
# Segmentation (mock the Transformers pipeline)
# ---------------------------------------------------------------------------

def test_segment_filters_by_area_and_removes_duplicates(monkeypatch):
    masks = []
    # Valid mask.
    first = np.zeros((40, 40), dtype=bool)
    first[5:20, 5:20] = True
    masks.append(first)
    # Near-identical mask should be suppressed by NMS.
    duplicate = first.copy()
    duplicate[5, 5] = False
    masks.append(duplicate)
    # Too small.
    tiny = np.zeros((40, 40), dtype=bool)
    tiny[0:2, 0:2] = True
    masks.append(tiny)
    # Too large.
    huge = np.ones((40, 40), dtype=bool)
    masks.append(huge)

    monkeypatch.setattr(
        sut, "pipeline",
        lambda *a, **kw: lambda *args, **kwargs: {
            "masks": masks, "scores": [0.9, 0.8, 0.7, 0.6]
        },
    )
    result = sut.segment(Image.new("RGB", (40, 40)), "fake-model", 8, "cpu")
    assert len(result) == 1
    assert result[0].name is None
    assert result[0].score == pytest.approx(0.9)
    np.testing.assert_array_equal(result[0].mask, first)


@pytest.mark.parametrize("error_type", [TypeError, ValueError])
def test_segment_retries_without_grid_option_on_compatibility_error(monkeypatch, error_type):
    calls = []

    def generator(*args, **kwargs):
        calls.append(kwargs)
        if "points_per_crop" in kwargs:
            raise error_type("unsupported option")
        return {"masks": [], "scores": []}

    monkeypatch.setattr(sut, "pipeline", lambda *a, **kw: generator)
    assert sut.segment(Image.new("RGB", (10, 10)), "fake-model", 4, "cpu") == []
    assert len(calls) == 2
    assert "points_per_crop" in calls[0]
    assert "points_per_crop" not in calls[1]


def test_load_vlm_cpu_loads_and_evaluates_model(monkeypatch):
    calls = []

    class FakeModel:
        def to(self, device):
            calls.append(("to", device))
            return self

        def eval(self):
            calls.append(("eval",))
            return self

    model = FakeModel()

    class FakeLoader:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            calls.append(("load", model_id, kwargs))
            return model

    monkeypatch.setattr(sut, "AutoModelForImageTextToText", FakeLoader)
    assert sut.load_vlm("test-model", "cpu") is model
    assert calls == [
        ("load", "test-model", {}),
        ("to", "cpu"),
        ("eval",),
    ]


def test_load_vlm_cuda_retries_with_legacy_torch_dtype(monkeypatch):
    calls = []

    class FakeModel:
        def to(self, device):
            calls.append(("to", device))
            return self

        def eval(self):
            calls.append(("eval",))
            return self

    model = FakeModel()

    class FakeLoader:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            calls.append(("load", model_id, kwargs))
            if "dtype" in kwargs:
                raise TypeError("use torch_dtype")
            return model

    monkeypatch.setattr(sut, "AutoModelForImageTextToText", FakeLoader)
    monkeypatch.setattr(sut.torch.cuda, "is_bf16_supported", lambda: False)
    assert sut.load_vlm("test-model", "cuda") is model
    assert calls == [
        ("load", "test-model", {"dtype": sut.torch.float16}),
        ("load", "test-model", {"torch_dtype": sut.torch.float16}),
        ("to", "cuda"),
        ("eval",),
    ]


# ---------------------------------------------------------------------------
# Naming backends
# ---------------------------------------------------------------------------

def test_vlm_ask_preserves_scene_aspect_ratio_and_decodes_answer():
    class FakeInputs(dict):
        def to(self, device):
            self.device = device
            return self

    class FakePixelValues:
        dtype = None

        def to(self, dtype):
            self.dtype = dtype
            return self

    class FakeProcessor:
        def __init__(self):
            self.pixel_values = FakePixelValues()

        def apply_chat_template(self, messages, **kwargs):
            self.messages = messages
            self.template_kwargs = kwargs
            return "rendered prompt"

        def __call__(self, text, images, return_tensors):
            self.text = text
            self.images = images
            self.return_tensors = return_tensors
            return FakeInputs(input_ids=sut.torch.tensor([[1, 2]]),
                              pixel_values=self.pixel_values)

        def batch_decode(self, sequences, **kwargs):
            self.sequences = sequences
            self.decode_kwargs = kwargs
            return ["a red cup"]

    class FakeModel:
        dtype = sut.torch.float16

        def generate(self, **kwargs):
            self.generate_kwargs = kwargs
            return sut.torch.tensor([[1, 2, 3]])

    namer = object.__new__(sut.VlmNamer)
    namer.device = "cpu"
    namer.processor = FakeProcessor()
    namer.model = FakeModel()
    context = Image.new("RGB", (100, 50), (255, 0, 0))
    crop = Image.new("RGB", (4, 2), (0, 0, 255))

    answer = namer._ask(context, crop, "Name the object.", max_new_tokens=5)

    padded_context, resized_crop = namer.processor.images
    assert answer == "a red cup"
    assert padded_context.size == (sut.VLM_CROP_SIZE, sut.VLM_CROP_SIZE)
    assert padded_context.getpixel((0, 0)) == (128, 128, 128)
    assert padded_context.getpixel((sut.VLM_CROP_SIZE // 2, sut.VLM_CROP_SIZE // 2)) == (255, 0, 0)
    assert resized_crop.size == (sut.VLM_CROP_SIZE, sut.VLM_CROP_SIZE)
    assert namer.processor.messages[0]["content"][-1]["text"] == "Name the object."
    assert namer.processor.template_kwargs == {
        "add_generation_prompt": True,
        "tokenize": False,
    }
    assert namer.processor.text == "rendered prompt"
    assert namer.processor.return_tensors == "pt"
    assert namer.processor.pixel_values.dtype is sut.torch.float16
    assert namer.model.generate_kwargs["max_new_tokens"] == 5
    assert namer.model.generate_kwargs["do_sample"] is False
    assert namer.processor.sequences.tolist() == [[3]]
    assert namer.processor.decode_kwargs == {"skip_special_tokens": True}


def test_vlm_verify_name_accepts_yes(monkeypatch):
    namer = object.__new__(sut.VlmNamer)
    monkeypatch.setattr(namer, "_ask", lambda *a, **kw: "Yes.")
    supported, answer = namer.verify_name(Image.new("RGB", (2, 2)),
                                          Image.new("RGB", (2, 2)), "cup")
    assert supported is True
    assert answer == "Yes."


@pytest.mark.parametrize(
    ("answer", "expected"),
    [("no", False), ("unclear", False), ("yes, but uncertainly", False)],
)
def test_vlm_verify_name_rejects_non_exact_affirmations(monkeypatch, answer, expected):
    namer = object.__new__(sut.VlmNamer)
    monkeypatch.setattr(namer, "_ask", lambda *a, **kw: answer)
    supported, returned = namer.verify_name(Image.new("RGB", (2, 2)),
                                             Image.new("RGB", (2, 2)), "cup")
    assert supported is expected
    assert returned == answer


def test_vlm_name_one_cleans_answer(monkeypatch):
    namer = object.__new__(sut.VlmNamer)
    monkeypatch.setattr(namer, "_ask", lambda *a, **kw: "The red mug.")
    assert namer.name_one(Image.new("RGB", (2, 2)), Image.new("RGB", (2, 2))) == "red mug"


def test_vlm_name_crops_verifies_and_rejects(monkeypatch):
    namer = object.__new__(sut.VlmNamer)
    namer.clip_diagnostic = None
    monkeypatch.setattr(namer, "name_one", lambda *a: "mug")
    monkeypatch.setattr(namer, "verify_name", lambda *a: (False, "no"))
    crops = [Image.new("RGB", (2, 2))]
    contexts = [Image.new("RGB", (2, 2))]
    assert namer.name_crops(crops, contexts) == [""]


def test_vlm_name_crops_keeps_verified_name(monkeypatch):
    namer = object.__new__(sut.VlmNamer)
    namer.clip_diagnostic = None
    monkeypatch.setattr(namer, "name_one", lambda *a: "mug")
    monkeypatch.setattr(namer, "verify_name", lambda *a: (True, "yes"))
    crops = [Image.new("RGB", (2, 2))]
    contexts = [Image.new("RGB", (2, 2))]
    assert namer.name_crops(crops, contexts) == ["mug"]


def test_vlm_name_crops_honors_cancellation():
    namer = object.__new__(sut.VlmNamer)
    namer.clip_diagnostic = None

    class Stop:
        def is_set(self):
            return True

    with pytest.raises(sut.Cancelled):
        namer.name_crops([Image.new("RGB", (2, 2))],
                         [Image.new("RGB", (2, 2))], Stop())


def test_clip_namer_batches_and_applies_confidence(monkeypatch):
    namer = object.__new__(sut.ClipNamer)
    namer.device = "cpu"
    namer.vocab = ["cup", "book"]
    namer.min_conf = 0.5
    namer.prompts = ["a photo of a cup", "a photo of a book", "a photo of a background"]
    batch_sizes = []

    class Inputs(dict):
        def to(self, device):
            return self

    class FakeLogits:
        def __init__(self, values):
            self.values = np.asarray(values, dtype=float)

        def detach(self):
            return self

        def float(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.values

    class FakeModel:
        def __call__(self, **kwargs):
            batch_size = kwargs["batch_size"]
            if batch_size == 8:
                logits = [[8.0, 0.0, -2.0]] * batch_size
            else:
                logits = [[0.0, 8.0, -2.0], [0.1, 0.0, -0.1]]
            return SimpleNamespace(logits_per_image=FakeLogits(logits))

    class FakeProcessor:
        def __call__(self, **kwargs):
            batch_size = len(kwargs["images"])
            batch_sizes.append(batch_size)
            return Inputs(batch_size=batch_size)

    namer.model = FakeModel()
    namer.processor = FakeProcessor()
    crops = [Image.new("RGB", (2, 2)) for _ in range(10)]
    assert namer.name_crops(crops) == ["cup"] * 8 + ["book", ""]
    assert batch_sizes == [8, 2]


def test_clip_namer_diagnose_name_ranks_and_returns_alternatives(monkeypatch):
    namer = object.__new__(sut.ClipNamer)
    namer.device = "cpu"
    namer.vocab = ["cup", "book", "lamp"]

    class Inputs(dict):
        def to(self, device):
            return self

    class FakeProcessor:
        def __call__(self, **kwargs):
            self.kwargs = kwargs
            return Inputs()

    class FakeModel:
        def __call__(self, **kwargs):
            return SimpleNamespace(logits_per_image=sut.torch.tensor(
                [[0.3, 0.9, 0.1, 0.4, 0.2, 0.05, 0.01, 0.0]]))

    namer.processor = FakeProcessor()
    namer.model = FakeModel()
    crop = Image.new("RGB", (2, 2))
    rank, total, alternatives = namer.diagnose_name(crop, "CUP")

    assert namer.processor.kwargs["text"][0] == "a photo of a CUP"
    assert namer.processor.kwargs["images"] is crop
    assert rank == 3
    assert total == 8
    assert alternatives == [
        ("book", pytest.approx(0.9)),
        ("background", pytest.approx(0.4)),
        ("blurry background", pytest.approx(0.2)),
    ]


def test_clip_namer_checks_cancellation_before_batch():
    namer = object.__new__(sut.ClipNamer)
    namer.device = "cpu"
    namer.vocab = ["cup"]
    namer.min_conf = 0.1
    namer.prompts = ["a photo of a cup"]

    class Stop:
        def is_set(self):
            return True

    with pytest.raises(sut.Cancelled):
        namer.name_crops([Image.new("RGB", (2, 2))], Stop())


# ---------------------------------------------------------------------------
# Quality matching and validation
# ---------------------------------------------------------------------------

def test_match_quality_masks_finds_best_non_conflicting_matches():
    target_a = np.zeros((8, 8), dtype=bool)
    target_a[:4, :4] = True
    target_b = np.zeros((8, 8), dtype=bool)
    target_b[4:, 4:] = True
    detections = [target_a.copy(), target_b.copy()]
    targets = [make_instance(target_a, "cup"), make_instance(target_b, "book")]

    matches = sut._match_quality_masks([0, 1], targets, detections, [0.8, 0.7])
    assert matches[0] == pytest.approx((1.0, 0.8))
    assert matches[1] == pytest.approx((1.0, 0.7))


def test_match_quality_masks_does_not_reuse_detection():
    target_a = np.zeros((8, 8), dtype=bool)
    target_a[:4, :4] = True
    target_b = target_a.copy()
    targets = [make_instance(target_a, "one"), make_instance(target_b, "two")]
    matches = sut._match_quality_masks([0, 1], targets, [target_a], [0.9])
    assert len(matches) == 1


def test_evaluate_naming_quality_rejects_empty_targets():
    with pytest.raises(ValueError, match="no named segments"):
        sut.evaluate_naming_quality(Image.new("RGB", (4, 4)), [], 1, "cpu")


def test_evaluate_naming_quality_rejects_invalid_run_count():
    target = make_instance(np.ones((4, 4), dtype=bool), "cup")
    with pytest.raises(ValueError, match="Run count"):
        sut.evaluate_naming_quality(Image.new("RGB", (4, 4)), [target], 0, "cpu")


def test_evaluate_naming_quality_rejects_no_usable_names():
    target = make_instance(np.ones((4, 4), dtype=bool), "")
    with pytest.raises(ValueError, match="no usable names"):
        sut.evaluate_naming_quality(Image.new("RGB", (4, 4)), [target], 1, "cpu")


def test_evaluate_naming_quality_scores_and_averages_mock_detections(monkeypatch):
    cup_left = np.zeros((4, 4), dtype=bool)
    cup_left[:2, :2] = True
    cup_right = np.zeros((4, 4), dtype=bool)
    cup_right[2:, 2:] = True
    book_mask = np.zeros((4, 4), dtype=bool)
    book_mask[:2, 2:] = True
    targets = [
        make_instance(cup_left, "cup"),
        make_instance(cup_right, "cup"),
        make_instance(book_mask, "book"),
    ]
    processors = []
    model_calls = []
    vision_calls = []

    class Inputs(dict):
        def to(self, device):
            return self

        def __getattr__(self, name):
            try:
                return self[name]
            except KeyError as exc:
                raise AttributeError(name) from exc

    class FakeProcessor:
        @classmethod
        def from_pretrained(cls, model_id):
            processor = cls()
            processor.model_id = model_id
            processor.image_input_count = 0
            processor.prompts = []
            processor.post_processed = []
            processors.append(processor)
            return processor

        def __call__(self, images=None, text=None, return_tensors=None):
            if images is not None:
                self.image_input_count += 1
                return Inputs(
                    pixel_values=sut.torch.zeros((1, 3, 4, 4)),
                    original_sizes=sut.torch.tensor([[4, 4]]),
                )
            self.prompts.append(text)
            return Inputs(prompt=text)

        def post_process_instance_segmentation(self, outputs, **kwargs):
            assert kwargs["threshold"] == 0.5
            assert kwargs["mask_threshold"] == 0.5
            assert kwargs["target_sizes"] == [[4, 4]]
            prompt = outputs.prompt
            self.post_processed.append(prompt)
            detections = {
                "cup": [(cup_left, 0.8), (cup_right, 0.6)],
                "a photo of cup": [(cup_left, 0.8), (cup_right, 0.6)],
                "a photo of book": [(book_mask, 0.5)],
            }.get(prompt, [])
            masks = [sut.torch.from_numpy(mask.copy()) for mask, _ in detections]
            scores = [sut.torch.tensor(score) for _, score in detections]
            return [{"masks": masks, "scores": scores}]

    class FakeModel:
        dtype = sut.torch.float32

        def get_vision_features(self, pixel_values):
            vision_calls.append(pixel_values.shape)
            return "vision-embeddings"

        def __call__(self, vision_embeds, prompt):
            model_calls.append((vision_embeds, prompt))
            return SimpleNamespace(prompt=prompt)

    model = FakeModel()
    monkeypatch.setattr(sut.base, "load_model",
                        lambda device, half, fast: model)
    monkeypatch.setattr(sut.base, "Sam3Processor", FakeProcessor)
    monkeypatch.setattr(sut.base, "sync", lambda device: None)

    report = sut.evaluate_naming_quality(
        Image.new("RGB", (4, 4)), targets, runs=2, device="cpu")

    assert processors[0].model_id == sut.base.SAM3_MODEL
    assert processors[0].image_input_count == 1
    assert vision_calls == [(1, 3, 4, 4)]
    assert [prompt for _, prompt in model_calls] == [
        "cup", "book", "a photo of cup", "a photo of book",
    ]
    assert report["prompt_templates"] == list(sut.QUALITY_PROMPT_TEMPLATES[:2])
    assert len(report["items"][0]["runs"]) == 2
    assert report["items"][0]["mean_iou"] == pytest.approx(1.0)
    assert report["items"][0]["mean_sam3_confidence"] == pytest.approx(0.8)
    assert report["items"][0]["mean_score"] == pytest.approx(80.0)
    assert report["items"][1]["mean_sam3_confidence"] == pytest.approx(0.6)
    assert report["items"][2]["runs"][0]["score"] == 0
    assert report["items"][2]["mean_iou"] == pytest.approx(0.5)
    assert report["items"][2]["mean_sam3_confidence"] == pytest.approx(0.25)
    assert report["items"][2]["mean_score"] == pytest.approx(25.0)


# ---------------------------------------------------------------------------
# Report and cache helpers
# ---------------------------------------------------------------------------

def test_save_quality_report_writes_metadata(tmp_path):
    source_image = tmp_path / "photo.jpg"
    source_image.write_bytes(b"test image bytes")
    report = {"runs": 1, "items": []}
    path = sut.save_quality_report(source_image, "custom.jpg", tmp_path / "cache", report)

    saved = json.loads(path.read_text(encoding="utf-8"))
    assert path.exists()
    assert saved["image_name"] == "custom.jpg"
    assert len(saved["image_sha1"]) == 16
    assert saved["runs"] == 1
    assert "saved_at" in saved


def test_save_quality_report_uses_path_name_when_image_name_missing(tmp_path):
    source_image = tmp_path / "photo.jpg"
    source_image.write_bytes(b"bytes")
    path = sut.save_quality_report(source_image, None, tmp_path / "cache", {})
    assert json.loads(path.read_text())["image_name"] == "photo.jpg"


def test_sha_is_order_independent_for_dicts():
    assert sut.sha({"a": 1, "b": 2}) == sut.sha({"b": 2, "a": 1})


def test_sha_changes_when_value_changes():
    assert sut.sha({"a": 1}) != sut.sha({"a": 2})


def test_names_path_replaces_cache_suffix(tmp_path):
    source = tmp_path / "segments_abc.sam3cache.npz"
    result = sut.names_path(source, "vlm123")
    assert result.name == "segments_abc.names_vlm123.json"


def test_load_names_restores_names(tmp_path):
    path = tmp_path / "names.json"
    path.write_text(json.dumps({"names": ["cup", "", None]}), encoding="utf-8")
    instances = [
        make_instance(np.ones((2, 2), dtype=bool)),
        make_instance(np.ones((2, 2), dtype=bool)),
        make_instance(np.ones((2, 2), dtype=bool)),
    ]
    sut.load_names(path, instances)
    assert [item.name for item in instances] == ["cup", "", None]


def test_load_names_ignores_wrong_length(tmp_path):
    path = tmp_path / "names.json"
    path.write_text(json.dumps({"names": ["only one"]}), encoding="utf-8")
    instances = [make_instance(np.ones((2, 2), dtype=bool)),
                 make_instance(np.ones((2, 2), dtype=bool))]
    sut.load_names(path, instances)
    assert [item.name for item in instances] == [None, None]


def test_load_names_ignores_invalid_file(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{not valid json", encoding="utf-8")
    instances = [make_instance(np.ones((2, 2), dtype=bool))]
    sut.load_names(path, instances)
    assert instances[0].name is None


def test_save_names_and_reload(tmp_path):
    path = tmp_path / "nested" / "names.json"
    instances = [
        make_instance(np.ones((2, 2), dtype=bool), "cup"),
        make_instance(np.zeros((2, 2), dtype=bool), ""),
    ]
    sut.save_names(path, instances)
    assert json.loads(path.read_text(encoding="utf-8")) == {"names": ["cup", ""]}
    restored = [make_instance(i.mask) for i in instances]
    sut.load_names(path, restored)
    assert [i.name for i in restored] == ["cup", ""]


def test_segment_cache_path_is_deterministic(tmp_path):
    image_path = tmp_path / "photo.jpg"
    image_path.write_bytes(b"same image")
    args = SimpleNamespace(sam_model="sam-model", grid=24)
    first_path, first_key = sut.segment_cache_path(tmp_path / "cache", image_path, args)
    second_path, second_key = sut.segment_cache_path(tmp_path / "cache", image_path, args)
    assert first_path == second_path
    assert first_key == second_key
    assert first_path.name.endswith(".sam3cache.npz")


def test_segment_cache_path_changes_with_grid(tmp_path):
    image_path = tmp_path / "photo.jpg"
    image_path.write_bytes(b"same image")
    args_a = SimpleNamespace(sam_model="sam-model", grid=16)
    args_b = SimpleNamespace(sam_model="sam-model", grid=32)
    assert sut.segment_cache_path(tmp_path, image_path, args_a)[1] != \
           sut.segment_cache_path(tmp_path, image_path, args_b)[1]


# ---------------------------------------------------------------------------
# Preparation orchestration
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("namer_kind", ["clip", "vlm"])
def test_prepare_uses_selected_namer_and_returns_opening_game(tmp_path, monkeypatch, namer_kind):
    image_path = tmp_path / "photo.jpg"
    Image.new("RGB", (16, 16), (90, 100, 110)).save(image_path)
    mask = np.zeros((16, 16), dtype=bool)
    mask[4:10, 4:10] = True
    instance = make_instance(mask)
    monkeypatch.setattr(sut, "pick_device", lambda: "cpu")
    monkeypatch.setattr(sut, "segment", lambda *args: [instance])
    monkeypatch.setattr(sut, "load_concepts", lambda *args: ["cup"])
    args = SimpleNamespace(
        no_cache=True, recompute=False, cache_dir=tmp_path,
        sam_model="fake-sam", grid=8, namer=namer_kind,
        groups="general", auto_top=2, scene_model="fake-scene",
        clip_model="fake-clip", min_confidence=0.2,
        clip_diagnostic=False, vlm_model=None, vlm_size="500M",
    )

    class FakeClipNamer:
        def __init__(self, clip_model, vocab, device, min_confidence):
            assert (clip_model, vocab, device, min_confidence) == (
                "fake-clip", ["cup"], "cpu", 0.2,
            )

        def name_crops(self, crops, stop=None):
            assert len(crops) == 1
            return ["cup"]

    class FakeVlmNamer:
        def __init__(self, model_id, device, clip_diagnostic):
            assert model_id == sut.VLM_MODELS["500M"]
            assert device == "cpu"
            assert clip_diagnostic is None

        def name_crops(self, crops, context_images, stop=None):
            assert len(crops) == len(context_images) == 1
            assert context_images[0].size == (16, 16)
            return ["cup"]

    monkeypatch.setattr(sut, "ClipNamer", FakeClipNamer)
    monkeypatch.setattr(sut, "VlmNamer", FakeVlmNamer)

    def pick_chosen(instances, namer, stop_event):
        namer(instances, stop_event)
        return list(instances)

    monkeypatch.setattr(sut.base, "pick_chosen", pick_chosen)

    image, instances, extras = sut.prepare(image_path, args)

    assert image.size == (16, 16)
    assert instances == [instance]
    assert instance.name == "cup"
    assert extras["first_game"] == [instance]
    assert callable(extras["namer"])


def test_prepare_loads_cached_segments_and_names(tmp_path, monkeypatch):
    image_path = tmp_path / "photo.jpg"
    Image.new("RGB", (16, 16)).save(image_path)
    segment_path = tmp_path / "segments.sam3cache.npz"
    segment_path.write_bytes(b"cached")
    names_file = tmp_path / "names.json"
    names_file.write_text('{"names":["cup"]}', encoding="utf-8")
    mask = np.zeros((16, 16), dtype=bool)
    mask[4:10, 4:10] = True
    instance = make_instance(mask, "stale")
    monkeypatch.setattr(sut, "pick_device", lambda: "cpu")
    monkeypatch.setattr(sut, "segment_cache_path",
                        lambda *args: (segment_path, "segment-key"))
    monkeypatch.setattr(sut.base, "load_cache", lambda path, shape: [instance])
    monkeypatch.setattr(sut, "names_path", lambda *args: names_file)
    monkeypatch.setattr(sut, "load_concepts", lambda *args: ["cup"])
    monkeypatch.setattr(sut, "segment",
                        lambda *args: pytest.fail("segment cache should avoid recomputation"))

    class FakeClipNamer:
        def __init__(self, *args):
            pass

    monkeypatch.setattr(sut, "ClipNamer", FakeClipNamer)

    def pick_chosen(instances, namer, stop_event):
        assert instances[0].name == "cup"
        return list(instances)

    monkeypatch.setattr(sut.base, "pick_chosen", pick_chosen)
    args = SimpleNamespace(
        no_cache=False, recompute=False, cache_dir=tmp_path,
        sam_model="fake-sam", grid=8, namer="clip",
        groups="general", auto_top=2, scene_model="fake-scene",
        clip_model="fake-clip", min_confidence=0.2,
    )

    _, instances, extras = sut.prepare(image_path, args)

    assert instances == [instance]
    assert instance.name == "cup"
    assert extras["first_game"] == [instance]


# ---------------------------------------------------------------------------
# Command-line parser
# --------------------------------------------------------------------------------------

def test_parser_defaults():
    args = sut.build_parser().parse_args([])
    assert args.namer == "vlm"
    assert args.vlm_size == sut.DEFAULT_VLM_SIZE
    assert args.grid == sut.DEFAULT_GRID
    assert args.sam_model == sut.SAM_MODEL
    assert args.groups == "auto"
    assert args.no_cache is False
    assert args.recompute is False
    assert args.overlay is False


def test_parser_accepts_clip_options():
    args = sut.build_parser().parse_args([
        "scene.jpg", "--namer", "clip", "--groups", "bathroom,toys",
        "--grid", "12", "--min-confidence", "0.3", "--no-cache", "--cheat",
    ])
    assert args.image == "scene.jpg"
    assert args.namer == "clip"
    assert args.groups == "bathroom,toys"
    assert args.grid == 12
    assert args.min_confidence == pytest.approx(0.3)
    assert args.no_cache is True
    assert args.overlay is True


def test_parser_rejects_unknown_namer():
    with pytest.raises(SystemExit):
        sut.build_parser().parse_args(["--namer", "unknown"])
