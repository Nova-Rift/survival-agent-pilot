"""Render recorded numerical simulation frames, never screenshots or inference.
Pillow is an optional rendering dependency; no external requests are made.
"""
import argparse
import colorsys
import hashlib
import json
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    raw=args.frames.read_bytes()
    if len(raw)>2*1024*1024:raise ValueError('Preview trace exceeds size limit')
    frames=json.loads(raw)
    if not isinstance(frames,list) or not 1<=len(frames)<=256:raise ValueError('Expected bounded recorded frames')
    images=[]
    font=ImageFont.load_default()
    for frame in frames:
        if set(frame)!={'tick','positions','alive','hunger','food'}:raise ValueError('Unexpected frame fields')
        if len(frame['positions'])!=8 or len(frame['alive'])!=8 or len(frame['hunger'])!=8:raise ValueError('Expected eight agents')
        if type(frame['tick']) is not int or not 0<=frame['tick']<=128:raise ValueError('Invalid tick')
        coords=[*frame['positions'],*frame['food']]
        if any(len(p)!=2 or any(type(v) is not int or not 0<=v<16 for v in p) for p in coords):raise ValueError('Invalid grid coordinate')
        image=Image.new('RGB',(400,402),'#f2f3eb');draw=ImageDraw.Draw(image)
        draw.text((16,10),'RECORDED SURVIVAL SIMULATION',fill='#20392b',font=font)
        draw.text((16,27),'Scripted greedy baseline | seed 0 | untrained',fill='#4d6154',font=font)
        left,top,cell=40,51,20
        draw.rectangle((left,top,left+320,top+320),fill='#1e3028')
        for n in range(17):
            draw.line((left+n*cell,top,left+n*cell,top+320),fill='#344b42')
            draw.line((left,top+n*cell,left+320,top+n*cell),fill='#344b42')
        for y,x in frame['food']:
            cx,cy=left+x*cell+10,top+y*cell+10
            draw.ellipse((cx-3,cy-3,cx+3,cy+3),fill='#96da8c')
        for i,(y,x) in enumerate(frame['positions']):
            if not frame['alive'][i]:continue
            color=tuple(int(c*255) for c in colorsys.hls_to_rgb(i/8,.68,.70))
            px,py=left+x*cell+2,top+y*cell+2
            draw.rectangle((px,py,px+15,py+15),fill=color)
            draw.text((px+4,py+2),str(i),fill='#16241c',font=font)
            hunger=frame['hunger'][i]
            if type(hunger) is not int or not 0<=hunger<=20:raise ValueError('Invalid hunger')
            draw.line((px,py+15,px+15*hunger/20,py+15),fill='white',width=1)
        alive=sum(bool(v) for v in frame['alive'])
        draw.text((16,382),f"Step {frame['tick']:3d}/128 | {alive}/8 alive | Green dots: food",fill='#20392b',font=font)
        images.append(image.convert('P',palette=Image.Palette.ADAPTIVE,colors=64))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    images[0].save(args.output,save_all=True,append_images=images[1:],duration=140,loop=0,optimize=True,disposal=2)
    provenance={'source':'Actual locally recorded simulation already shown on clarkwhitehead.com',
                'policy':'Scripted observation-limited greedy baseline; not a trained model',
                'seed':0,'source_frames':len(frames),'source_sha256':hashlib.sha256(raw).hexdigest(),
                'dimensions':[400,402],'frame_duration_ms':140,'asset_bytes':args.output.stat().st_size,
                'model_inference':False,'screenshot':False}
    args.output.with_name('preview-provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps(provenance))

if __name__=='__main__':main()
