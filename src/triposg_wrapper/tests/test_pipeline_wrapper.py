from types import SimpleNamespace

import PIL.Image
import pytest
import torch

from src.triposg_wrapper.triposg_pipeline_wrapper import TripoSGPipelineWrapper


class FakeProcessor:
    def __call__(self, images, return_tensors="pt"):
        batch = len(images) if isinstance(images, list) else 1
        return SimpleNamespace(pixel_values=torch.ones(batch, 3, 8, 8))


class FakeTokenizer:
    model_max_length = 4

    def __call__(self, prompts, **kwargs):
        return {"input_ids": torch.ones(len(prompts), 4, dtype=torch.long)}


class FakeEncoder(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.width = width

    def forward(self, values):
        return SimpleNamespace(
            last_hidden_state=torch.ones(values.shape[0], 4, self.width)
        )

    @property
    def device(self):
        return self.weight.device


class FakeTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.config = SimpleNamespace(in_channels=8)
        self.calls = []

    def forward(self, latents, timestep, **kwargs):
        self.calls.append((latents.shape, timestep.dtype, kwargs))
        return (torch.zeros_like(latents),)

    @property
    def device(self):
        return self.weight.device


class FakeScheduler:
    order = 1

    def set_timesteps(self, num_inference_steps, device=None):
        self.timesteps = torch.arange(num_inference_steps - 1, -1, -1, device=device)

    def step(self, noise_pred, timestep, latents, return_dict=False):
        return (latents,)


class FakeVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def decode(self, latents, sampled_points):
        return SimpleNamespace(sample=torch.zeros((*sampled_points.shape[:-1], 1)))

    @property
    def device(self):
        return self.weight.device


def make_pipe(*, scribble):
    return TripoSGPipelineWrapper(
        vae=FakeVAE(),
        transformer=FakeTransformer(),
        scheduler=FakeScheduler(),
        image_encoder_dinov2=FakeEncoder(6),
        feature_extractor_dinov2=FakeProcessor(),
        tokenizer=FakeTokenizer() if scribble else None,
        text_encoder=FakeEncoder(5) if scribble else None,
    )


def test_text_latents_use_dual_conditioning_and_scribble_scale_zero():
    pipe = make_pipe(scribble=True)
    latents = pipe.text2latent("a chair", num_inference_steps=2)

    assert latents.shape == (1, 512, 8)
    _, timestep_dtype, call = pipe.transformer.calls[0]
    assert timestep_dtype == latents.dtype
    assert call["encoder_hidden_states"].shape == (1, 4, 5)
    assert call["encoder_hidden_states_2"].shape == (1, 4, 6)
    assert call["attention_kwargs"] == {
        "cross_attention_scale": 1.0,
        "cross_attention_2_scale": 0.0,
    }


@pytest.mark.parametrize(
    ("prompt", "prompt_scale"), [("", 0.0), ("a winged cat", 1.0)]
)
def test_scribble_latents_support_scribble_only_and_combined(prompt, prompt_scale):
    pipe = make_pipe(scribble=True)
    image = PIL.Image.new("RGB", (16, 16), "white")
    latents = pipe.scribble2latent(image, prompt, num_inference_steps=1)

    assert latents.shape == (1, 512, 8)
    attention = pipe.transformer.calls[0][2]["attention_kwargs"]
    assert attention["cross_attention_scale"] == prompt_scale
    assert attention["cross_attention_2_scale"] == 0.4


def test_prompt_batch_must_match_scribble_batch():
    pipe = make_pipe(scribble=True)
    images = [PIL.Image.new("RGB", (8, 8)) for _ in range(2)]
    with pytest.raises(ValueError, match="image batch of size 2"):
        pipe.scribble2latent(images, ["only one"], num_inference_steps=1)


def test_checkpoint_specific_methods_fail_clearly():
    image_pipe = make_pipe(scribble=False)
    image = PIL.Image.new("RGB", (8, 8))
    with pytest.raises(ValueError, match="TripoSG-scribble"):
        image_pipe.text2latent("a chair")

    scribble_pipe = make_pipe(scribble=True)
    with pytest.raises(ValueError, match="image-only"):
        scribble_pipe.img2latent(image)


def test_image_latents_keep_original_defaults_and_single_condition():
    pipe = make_pipe(scribble=False)
    latents = pipe.img2latent(
        PIL.Image.new("RGB", (8, 8)), num_inference_steps=1
    )

    assert latents.shape == (1, 2048, 8)
    call = pipe.transformer.calls[0][2]
    assert "encoder_hidden_states_2" not in call
    assert call["encoder_hidden_states"].shape == (2, 4, 6)


def test_generated_latents_feed_geometric_function_and_reject_batches():
    pipe = make_pipe(scribble=True)
    latents = pipe.text2latent("a chair", num_inference_steps=1)
    sdf = pipe.make_geometric_func(latents)(torch.zeros(1, 3, 3))

    assert sdf.shape == (1, 3, 1)
    assert torch.count_nonzero(sdf) == 0
    with pytest.raises(ValueError, match="exactly one shape"):
        pipe.make_geometric_func(latents.repeat(2, 1, 1))


def test_generated_latents_feed_sdf_and_octree(monkeypatch):
    import src.triposg_wrapper.triposg_pipeline_wrapper as wrapper_module

    pipe = make_pipe(scribble=True)
    latents = pipe.text2latent("a chair", num_inference_steps=1)
    grid = -torch.ones(3, 3, 3)
    anchors = torch.ones(2, 2, 2, dtype=torch.int8)

    def fake_extract(geometric_func, **kwargs):
        assert geometric_func(torch.zeros(1, 2, 3)).shape == (1, 2, 1)
        assert kwargs["device"] == torch.device("cpu")
        return grid, anchors

    monkeypatch.setattr(wrapper_module, "hierarchical_extraction", fake_extract)
    monkeypatch.setattr(
        wrapper_module, "filter_largest_interior_component", lambda sdf, **kwargs: sdf
    )
    monkeypatch.setattr(
        wrapper_module,
        "sdf2octree",
        lambda **kwargs: (
            torch.zeros(8, 3),
            torch.zeros(8),
            torch.zeros(1, 8, dtype=torch.long),
            None,
            None,
        ),
    )

    sdf, returned_anchors = pipe.latent2sdf(latents, init_octree_depth=1, octree_depth=1)
    assert sdf is grid and returned_anchors is anchors
    octree = pipe.latent2octree(latents, (-1, -1, -1, 1, 1, 1), 1, 1, 2.0)
    assert octree["grid_sdf"] is grid
    assert octree["nodes"].shape == (8, 3)
