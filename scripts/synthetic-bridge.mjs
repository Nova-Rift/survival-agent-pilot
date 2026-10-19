// CPU training bridge to the exact published environment; no reimplemented rules.
import {createInterface} from 'node:readline';
import {resolve} from 'node:path';
import {pathToFileURL} from 'node:url';
const directory = resolve(process.argv[2]);
const {SyntheticWorld, SYNTHETIC_VERSION} = await import(pathToFileURL(resolve(directory, 'synthetic.mjs')));
const {createScriptedController} = await import(pathToFileURL(resolve(directory, 'scripted.mjs')));
let world, controller;
function snapshot(extra = {}) {
  return {version:SYNTHETIC_VERSION, config:world.config,
    observations:world.observe().map(x => Array.from(x)), frame:world.frame(),
    done:world.done, ...extra};
}
for await (const line of createInterface({input:process.stdin})) {
  try {
    const request = JSON.parse(line);
    let response;
    if (request.op === 'reset') {
      world = new SyntheticWorld({seed:request.seed});
      controller = createScriptedController({...world.config, seed:request.seed});
      response = snapshot();
    } else if (request.op === 'step') {
      const actions = request.scripted ? controller.actions(world.observe()) : request.actions;
      const active = world.frame().alive;
      const result = world.step(actions);
      response = snapshot({actions, active, rewards:Array.from(result.rewards), info:result.info});
    } else throw new Error('Unknown operation');
    process.stdout.write(JSON.stringify(response)+'\n');
  } catch (error) {
    process.stdout.write(JSON.stringify({error:String(error)})+'\n');
  }
}
