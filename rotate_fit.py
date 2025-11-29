#!/usr/bin/env python3

from torchvision.transforms.v2.functional import InterpolationMode
from torchvision.transforms.v2 import RandomRotation

from math import cos, sin, pi


class RandomRotationFit(RandomRotation):
  """Rotate the input by angle and crop with border radius.

  Use like :class:`RandomRotation`. Rotate around center of image is always assumed.
  """

  def __init__(
    self,
    degrees: Union[numbers.Number, Sequence],
    interpolation: Union[InterpolationMode, int] = InterpolationMode.NEAREST,
    fill: Union[_FillType, dict[Union[type, str], _FillType]] = 0,
  ) -> None:
    super().__init__(
      degrees=degrees,
      interpolation=interpolation,
      expand=True,
      center=None,
      fill=fill
    )

  def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
    if params['angle'] == 0.:
      return inpt
    _, H, W = inpt.shape
    r = (H if H < W else W) // 2
    I = super().transform(inpt, params)
    deg = params['angle'] * (pi / 180)
    _, h, w = I.shape
    ww, hh = [], []
    for x,y in [
      (-(W//2) + r, -(H//2) + r),
      (W - r - W//2, H - r - H//2)
    ]:
      if params['angle'] == 90.:
        x1, y1 = -y, x
      if params['angle'] == 180.:
        x1, y1 = -x, -y
      if params['angle'] == 270.:
        x1, y1 = y, -x
      else:
        x1, y1 = x*cos(deg) - y*sin(deg), x*sin(deg) + y*cos(deg)
      ww.append(int(w//2 + x1))
      hh.append(int(h//2 - y1))
    return I[:, max([0, min(hh)-r]):max(hh)+r, max([0, min(ww)-r]):max(ww)+r]

