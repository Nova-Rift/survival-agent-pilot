"""Optional local plain-text transport; no language model or reward shaping.

Kept separate so the original no-communication baseline remains unchanged.
World coordinates are used by delivery infrastructure, never put in envelopes.
"""
from dataclasses import asdict, dataclass
from collections.abc import Mapping

@dataclass(frozen=True)
class Message:
    sender: int
    sent_tick: int
    text: str

class LocalCommunication:
    def __init__(self, world, radius=2, max_chars=256):
        if type(radius) is not int or radius < 0 or type(max_chars) is not int or not 1 <= max_chars <= 256:
            raise ValueError('Invalid bounded communication configuration')
        self.world=world
        self.radius=radius
        self.max_chars=max_chars
        self._inboxes=[tuple() for _ in range(world.config.agents)]

    def inbox(self, agent):
        if type(agent) is not int or not 0 <= agent < self.world.config.agents:
            raise ValueError('Unknown agent')
        return self._inboxes[agent]

    def step(self, actions, utterances=None):
        """At most one message per sender per step. Range is Chebyshev distance
        at the START of the transition (same square geometry as observations).
        Deliver at its end only if both parties survive. No self/global delivery,
        no forwarding, and inboxes expire at the next step. Text has no authority
        to execute code, modify world state, or reveal another agent's abilities.
        """
        utterances={} if utterances is None else utterances
        if not isinstance(utterances,Mapping):raise ValueError('Expected sender-to-text mapping')
        for sender,text in utterances.items():
            if type(sender) is not int or not 0 <= sender < self.world.config.agents:
                raise ValueError('Unknown sender')
            if not self.world.alive[sender]:raise ValueError('Dead agents cannot send')
            if not isinstance(text,str) or not text.strip() or len(text)>self.max_chars:
                raise ValueError('Expected nonblank bounded plain text')
        positions=self.world.positions.copy();tick=self.world.tick
        result=self.world.step(actions)
        inboxes=[[] for _ in self._inboxes]
        for sender in sorted(utterances):
            if not self.world.alive[sender]:continue
            envelope=Message(sender,tick,utterances[sender])
            for recipient in range(self.world.config.agents):
                if recipient==sender or not self.world.alive[recipient]:continue
                if max(abs(positions[recipient]-positions[sender]))<=self.radius:
                    inboxes[recipient].append(envelope)
        self._inboxes=[tuple(messages) for messages in inboxes]
        return result

    def state(self):
        return {'format':1,'radius':self.radius,'max_chars':self.max_chars,
                'world':self.world.state(),
                'inboxes':[[asdict(m) for m in inbox] for inbox in self._inboxes]}

    def load_state(self,state):
        if state['format']!=1 or state['radius']!=self.radius or state['max_chars']!=self.max_chars:
            raise ValueError('Communication configuration mismatch')
        if len(state['inboxes'])!=self.world.config.agents:raise ValueError('Population mismatch')
        inboxes=[tuple(Message(**m) for m in inbox) for inbox in state['inboxes']]
        self.world.load_state(state['world']);self._inboxes=inboxes
