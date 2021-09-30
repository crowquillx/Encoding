import lvsfunc as lvf 
import vardefunc as vdf
import kagefunc as kgf
import fvsfunc as fvf
import havsfunc as hav
import mvsfunc as mvf
from functools import partial
import debandshit as dbs
import havsfunc as haf
import vapoursynth as vs
from vsutil import depth, get_depth, get_y, join, plane, split

core = vs.core

core.max_cache_size = 16384 

#shader = r"C:\Users\tanne\mpv\shaders\FSRCNN_x2_r2_32-0-2.glsl"

opstart = []
str_deb = []
if opstart:
    str_deb = str_deb + [(opstart+1648, opstart+1680)]
descale_args = dict(height=[828])
src = lvf.src("01.m2ts")
src = depth(src, 32)

#descale to 828p and rescale to 1080
descale1 = lvf.scale.descale(src, threshold=0.00225, **descale_args)
descale_m = lvf.scale.descale(src, show_mask=True, **descale_args)
rescale = vdf.scale.nnedi3_upscale(descale1)
resize = rescale.resize.Bicubic(1920, 1080)
rescale2 = core.std.MaskedMerge(resize, src, descale_m)

#become 16 bit
rescale2 = depth(rescale2,16)

#mask lines and apply aa
lmask = core.tcanny.TCanny(rescale2).std.Inflate()
aa = lvf.sraa(rescale2, rfactor=1.8, downscaler=lvf.kernels.Bicubic().scale)
aa = core.std.MaskedMerge(rescale2, aa, lmask)

#mask detail and apply debanding
detail_mask = lvf.mask.detail_mask(aa, sigma=False, brz_a=0.025, brz_b=0.045)
deband = dbs.placebo_deband(aa)
deband = core.std.MaskedMerge(deband, aa, detail_mask)

grained = kgf.adaptive_grain(deband,0.3,False,12,False)
#exit
final = depth(grained,10)
final.set_output()
