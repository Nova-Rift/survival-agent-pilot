"""One explicitly authorized pinned official artifact download. No credentials.
Not invoked by the backend, tests or simulation runner. No executable code fetched.
"""
import hashlib
import survival_agent
import json
import os
from pathlib import Path
import shutil
import urllib.request

REPO='Qwen/Qwen2.5-0.5B-Instruct'
REVISION='7ae557604adf67be50417f59c2c2f167def9a775'
DEST=Path(survival_agent.__file__).parent.parent/'models'/'qwen2.5-0.5b-instruct'
ALLOWED={'config.json','generation_config.json','tokenizer.json','tokenizer_config.json',
         'vocab.json','merges.txt','special_tokens_map.json','LICENSE','model.safetensors'}

def main():
    if shutil.disk_usage(DEST.parent if DEST.parent.exists() else DEST.parent.parent).free<4*2**30:
        raise RuntimeError('Need at least 4 GiB free; no download started')
    url=f'https://huggingface.co/api/models/{REPO}/revision/{REVISION}?blobs=true'
    with urllib.request.urlopen(url,timeout=30) as response:metadata=json.load(response)
    if metadata['sha']!=REVISION:raise RuntimeError('Revision mismatch')
    files=[s for s in metadata['siblings'] if s['rfilename'] in ALLOWED]
    if not {'model.safetensors','LICENSE','config.json','tokenizer.json','tokenizer_config.json'}<={s['rfilename'] for s in files}:
        raise RuntimeError('Required safe files unavailable')
    DEST.mkdir(parents=True,exist_ok=True)
    manifest={'repository':REPO,'revision':REVISION,'license':'Apache-2.0 per official model card',
              'source':'Official Hugging Face repository; no authentication headers or tokens','files':[]}
    for spec in sorted(files,key=lambda s:s['rfilename']):
        name=spec['rfilename'];target=DEST/name;expected=spec.get('lfs',{}).get('sha256')
        size=spec.get('size');part=target.with_suffix(target.suffix+'.part')
        if target.exists():raise RuntimeError(f'{name} already exists; will not overwrite a retained model')
        digest=hashlib.sha256();count=0
        print(f'Downloading {name} ({size} bytes)',flush=True)
        with urllib.request.urlopen(f'https://huggingface.co/{REPO}/resolve/{REVISION}/{name}',timeout=60) as response,part.open('wb') as output:
            while True:
                chunk=response.read(2**20)
                if not chunk:break
                output.write(chunk);digest.update(chunk);count+=len(chunk)
        checksum=digest.hexdigest()
        if size is not None and count!=size:raise RuntimeError(f'Size mismatch for {name}')
        if expected and checksum!=expected:raise RuntimeError(f'Checksum mismatch for {name}')
        os.replace(part,target)
        manifest['files'].append({'name':name,'bytes':count,'sha256':checksum,
                                  'repository_lfs_sha256_verified':bool(expected)})
    (DEST/'model-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    # Copy only small provenance information into Git; weights remain ignored.
    provenance_dir=Path(__file__).parent.parent/'artifacts';provenance_dir.mkdir(exist_ok=True)
    (provenance_dir/'qwen-model-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Pinned model files saved with SHA256 manifest',flush=True)

if __name__=='__main__':main()
