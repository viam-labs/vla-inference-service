"""Assert `_resize_with_pad` reproduces upstream's geometry exactly.

`ObservationBuilder._resize_with_pad` is a hand port of lerobot's
`resize_with_pad` (`lerobot/policies/common/vla_utils.py:219`) onto PIL, so
it carries the same risk the `ActionQueue` port does and gets the same
treatment: run both on identical inputs and compare. Requires lerobot:
`uv sync --extra lerobot`.

Scope note -- this compares *geometry*, not pixels. PIL's `BILINEAR` is
antialiased on downscale and `F.interpolate(mode="bilinear")` is not, so
exact pixel values legitimately differ (in the port's favour, for
downscaling, which is the case that actually matters here). What must agree
is the part a wrong port would silently break: the output shape, and where
the content lands inside it -- which side the padding goes on and how many
rows/columns of it there are. A solid white source on a black pad makes that
directly measurable as a bounding box.
"""

import numpy as np
import pytest

pytestmark = pytest.mark.differential

torch = pytest.importorskip("torch")
upstream_mod = pytest.importorskip("lerobot.policies.common.vla_utils")

from vla.controller.observation import ObservationBuilder

# (source_h, source_w, target_h, target_w). Chosen to cover every branch of
# the padding math, not just the common one: landscape pads in height only
# (pad_w == 0), portrait pads in width only (pad_h == 0) -- the case a
# square target can never exercise -- non-square targets pad in both,
# equal-aspect pads in neither, and the odd sizes pin down the `int()`
# truncation in `ratio` so an off-by-one cannot slip through.
CASES = [
    (180, 320, 256, 256),  # landscape -> square: pads top
    (320, 180, 256, 256),  # portrait  -> square: pads left
    (480, 640, 256, 320),  # landscape -> landscape, same aspect: no padding
    (480, 640, 512, 512),  # the real camera case, upscaled
    (64, 64, 256, 256),  # square upscale
    (224, 224, 224, 224),  # already at target: identity
    (7, 1000, 256, 256),  # extreme wide
    (1000, 7, 256, 256),  # extreme tall
    (223, 224, 224, 224),  # off-by-one, needs upscale in one axis
    (225, 224, 224, 224),  # off-by-one, needs downscale in one axis
    (100, 100, 256, 320),  # square source, non-square target
    (33, 77, 128, 128),  # odd sizes both axes
]


def _content_bbox(arr: np.ndarray) -> tuple[int, int, int, int]:
    """Bounding box of the non-black region as (top, bottom, left, right).

    The source is solid white and the pad is black, so this is exactly
    "where did the resized image land inside the padded canvas".
    """
    nonzero = arr.max(axis=2) > 0
    rows = np.flatnonzero(nonzero.any(axis=1))
    cols = np.flatnonzero(nonzero.any(axis=0))
    return int(rows[0]), int(rows[-1]), int(cols[0]), int(cols[-1])


@pytest.mark.parametrize(("src_h", "src_w", "tgt_h", "tgt_w"), CASES)
def test_resize_with_pad_matches_upstream_geometry(src_h, src_w, tgt_h, tgt_w):
    source = np.full((src_h, src_w, 3), 255, dtype=np.uint8)

    ours = ObservationBuilder._resize_with_pad(source, tgt_h, tgt_w)

    # Upstream takes (b, c, h, w) float and pads with `pad_value`; smolvla's
    # prepare_images passes pad_value=0, which is the black this port uses.
    t = torch.from_numpy(source).permute(2, 0, 1).unsqueeze(0).float().div_(255.0)
    theirs_t = upstream_mod.resize_with_pad(t, tgt_h, tgt_w, pad_value=0)
    theirs = (
        theirs_t.squeeze(0).permute(1, 2, 0).mul_(255.0).round().clamp_(0, 255).to(torch.uint8).numpy()
    )

    assert ours.shape == (tgt_h, tgt_w, 3)
    assert ours.shape == theirs.shape
    assert _content_bbox(ours) == _content_bbox(theirs)


@pytest.mark.parametrize(("src_h", "src_w", "tgt_h", "tgt_w"), CASES)
def test_padding_is_never_on_the_bottom_or_right(src_h, src_w, tgt_h, tgt_w):
    """The direction half of upstream's convention, asserted independently.

    `resize_with_pad`'s docstring is explicit that smolvla/xvla pad on the
    LEFT and TOP (the centered variant at vla_utils.py:142 is openpi's).
    Getting this backwards still produces a correctly-shaped,
    correctly-aspected image, so only an assertion on *which* edge the
    content reaches can catch it -- and it matters because this pad composes
    with smolvla's own: content must stay bottom-right through both.
    """
    source = np.full((src_h, src_w, 3), 255, dtype=np.uint8)
    ours = ObservationBuilder._resize_with_pad(source, tgt_h, tgt_w)
    top, bottom, left, right = _content_bbox(ours)
    assert bottom == tgt_h - 1, "content must reach the bottom edge; padding goes on top"
    assert right == tgt_w - 1, "content must reach the right edge; padding goes on the left"
    assert np.all(ours[:top] == 0), "the top padding band must be black"
    assert np.all(ours[:, :left] == 0), "the left padding band must be black"


# ---------------------------------------------------------------------------
# image_fit="stretch_bicubic" vs EVO1's own resize.
#
# `_batched_resize_01` (lerobot/policies/evo1/internvl3_embedder.py) is what
# actually runs on every frame inside an EVO1 policy. Its docstring states it
# exists to mirror InternVL3's reference PIL preprocessing -- `Image.resize`
# with the default (bicubic) resampler -- so the claim under test is that the
# controller's PIL bicubic reproduces it, and that the pre-existing "stretch"
# (BILINEAR) does not.
#
# This runs on the *controller* side rather than testing lerobot: the
# controller resizes a live camera frame onto the checkpoint's declared shape
# before the policy ever sees it, so a resampler mismatch here compounds with
# the policy's own resize into train/inference skew nothing downstream reports.
# ---------------------------------------------------------------------------

# Frames are random noise on purpose: it is the adversarial input for a
# resampler comparison. Natural images agree far more closely, so a tolerance
# that holds here holds everywhere that matters.
RESAMPLE_CASES = [
    (480, 640, 448),  # a 4:3 camera onto EVO1's native 448
    (1080, 1920, 448),  # heavy downscale
    (224, 224, 448),  # upscale from a square
    (300, 300, 448),  # non-power-of-two square
]

# Mean absolute difference of 0.30/255 across a whole frame. Empirically PIL
# and torchvision agree to 0.13-0.29 on random noise; BILINEAR is 3.3-13.1
# away, so this threshold separates "the same resampler" from "a different
# one" by an order of magnitude and is not a fudge factor.
_MEAN_TOLERANCE = 0.30


def _evo1_resize(frame: np.ndarray, image_size: int) -> np.ndarray:
    """Exactly what `_batched_resize_01` does, on one frame."""
    import torchvision.transforms.functional as tvf
    from torchvision.transforms.functional import InterpolationMode

    images = torch.from_numpy(frame).permute(2, 0, 1).float().div_(255.0).unsqueeze(0)
    pixels_u8 = (images * 255.0).clamp(0, 255).to(torch.uint8)
    resized = tvf.resize(
        pixels_u8, [image_size, image_size], interpolation=InterpolationMode.BICUBIC, antialias=True
    )
    return resized.squeeze(0).permute(1, 2, 0).numpy()


def _fit(frame: np.ndarray, image_fit: str, size: int) -> np.ndarray:
    from tests.fakes import FakeArm
    from vla.controller.gripper import make_gripper_adapter
    from vla.controller.units import UnitSegment, VectorUnits
    from vla.wire import decode_image

    builder = ObservationBuilder(
        cameras={},
        arm=FakeArm(),
        gripper=make_gripper_adapter({"type": "none"}, {}),
        state_joint_indices=[],
        state_units=VectorUnits((UnitSegment(3, "millimeters"), UnitSegment(6, "unitless"))),
        image_sizes={"k": (size, size)},
        image_encoding="raw",
        image_fit=image_fit,
        action_space="delta-ee",
    )
    return decode_image(builder._encode("k", frame))


@pytest.mark.parametrize(("src_h", "src_w", "size"), RESAMPLE_CASES)
def test_stretch_bicubic_reproduces_evo1s_resize(src_h, src_w, size):
    pytest.importorskip("lerobot.policies.evo1.internvl3_embedder")
    frame = np.random.default_rng(src_h * src_w).integers(
        0, 256, (src_h, src_w, 3), dtype=np.uint8
    )

    ours = _fit(frame, "stretch_bicubic", size)
    theirs = _evo1_resize(frame, size)

    assert ours.shape == theirs.shape == (size, size, 3)
    mean_error = float(np.abs(ours.astype(np.int32) - theirs.astype(np.int32)).mean())
    assert mean_error < _MEAN_TOLERANCE, (
        f"stretch_bicubic diverged from EVO1's resize by {mean_error:.4f}/255; "
        "the controller and the policy are no longer using the same resampler"
    )


@pytest.mark.parametrize(("src_h", "src_w", "size"), RESAMPLE_CASES)
def test_the_bilinear_stretch_would_not_have_passed(src_h, src_w, size):
    """The half that makes the test above mean something.

    Without this, `_MEAN_TOLERANCE` could be loose enough to accept any
    resampler at all and nobody would know. This pins that the pre-existing
    "stretch" -- the obvious thing to reuse -- fails the same threshold by an
    order of magnitude, which is why a third fit exists.
    """
    pytest.importorskip("lerobot.policies.evo1.internvl3_embedder")
    frame = np.random.default_rng(src_h * src_w).integers(
        0, 256, (src_h, src_w, 3), dtype=np.uint8
    )

    bilinear = _fit(frame, "stretch", size)
    theirs = _evo1_resize(frame, size)

    mean_error = float(np.abs(bilinear.astype(np.int32) - theirs.astype(np.int32)).mean())
    assert mean_error > _MEAN_TOLERANCE * 5, (
        f"BILINEAR is now within {mean_error:.4f}/255 of EVO1's resize; if that is "
        "genuinely true, `stretch_bicubic` no longer earns its place"
    )
