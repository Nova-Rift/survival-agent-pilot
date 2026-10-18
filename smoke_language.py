"""One short CPU inference for the selected local pretrained model.
No downloads, training, tools, APIs or world rollout. Run with an external timeout.
"""
from dataclasses import replace
from datetime import datetime,timezone
import json
from pathlib import Path
import resource
import time
import torch
from simulator import SurvivalWorld
from language_backend import LocalLanguageBackend, read_config, parse_decision, INSTRUCTION, local_view

def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    start=time.perf_counter()
    backend=LocalLanguageBackend(replace(read_config(Path(__file__).parent/'configs'/'pretrained.json'),max_new_tokens=32))
    loaded=time.perf_counter()
    prompt=INSTRUCTION+'\n'+json.dumps({'your_past_turns':[],'current':{'view':local_view(SurvivalWorld(0).observe()[0]),'received_messages':[]}},separators=(',',':'))
    output=backend.generate(prompt,seed=0)
    finished=time.perf_counter()
    report={'kind':'Actual selected-model local CPU inference smoke; not training or survival evidence',
            'recorded_utc':datetime.now(timezone.utc).isoformat(),'provenance':backend.provenance,
            'load_seconds':loaded-start,'generation_seconds':finished-loaded,
            'body_seconds':finished-start,'peak_process_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            'cpu_threads':torch.get_num_threads(),'maximum_generated_tokens':32,
            'generation_text':output,'parsed_decision':parse_decision(output),
            'api_calls':0,'optimizer_updates':0,'remote_compute':False,'cost_usd':0}
    path=Path(__file__).parent/'artifacts'/'qwen-cpu-smoke.json'
    path.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
