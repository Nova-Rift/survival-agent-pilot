from dataclasses import asdict
import unittest
import numpy as np
from simulator import Config, SurvivalWorld
from communication import LocalCommunication

class CommunicationTests(unittest.TestCase):
    def setup_channel(self):
        w=SurvivalWorld(1,Config(size=8,agents=4,initial_food=0,food_per_step=0))
        w.positions[:]=[[2,2],[2,4],[4,4],[7,7]]
        return LocalCommunication(w)

    def test_range_boundary_private_delivery_and_no_capability_metadata(self):
        c=self.setup_channel();c.step([0]*4,{0:'Is there food nearby?'})
        self.assertEqual(c.inbox(0),());self.assertEqual(c.inbox(3),())
        for i in [1,2]:
            self.assertEqual(len(c.inbox(i)),1)
            self.assertEqual(asdict(c.inbox(i)[0]),{'sender':0,'sent_tick':0,'text':'Is there food nearby?'})
        with self.assertRaises(AttributeError):c.inbox(1)[0].text='modified'

    def test_start_position_range_and_expiring_inboxes(self):
        c=self.setup_channel();c.world.positions[1]=[2,5]
        c.step([0,4,0,0],{0:'hello'})
        self.assertEqual(c.inbox(1),()) # Moved into range too late this step.
        c.step([0]*4,{0:'now nearby'});self.assertEqual(c.inbox(1)[0].text,'now nearby')
        c.step([0]*4);self.assertTrue(all(c.inbox(i)==() for i in range(4)))

    def test_dead_sender_or_recipient_and_invalid_messages(self):
        c=self.setup_channel();c.world.hunger[1]=1
        c.step([0]*4,{0:'hello',1:'last words'})
        self.assertEqual(c.inbox(1),());self.assertTrue(all(m.sender!=1 for m in c.inbox(0)))
        for messages in [{1:'dead'}, {99:'unknown'}, {0:'x'*257}, {0:'   '}, {0:5}, {True:'bad identity'}]:
            tick=c.world.tick
            with self.assertRaises(ValueError):c.step([0]*4,messages)
            self.assertEqual(c.world.tick,tick)

    def test_messages_do_not_change_world_or_rewards(self):
        c=self.setup_channel();other=self.setup_channel()
        r1=c.step([0]*4,{0:'share food and cooperate'})
        r2=other.step([0]*4)
        self.assertEqual(c.world.frame(),other.world.frame())
        np.testing.assert_array_equal(r1[0],r2[0]);np.testing.assert_array_equal(r1[1],r2[1])

    def test_transport_state_restore_preserves_next_delivery(self):
        c=self.setup_channel();c.step([0]*4,{0:'hello'})
        restored=self.setup_channel();restored.load_state(c.state())
        self.assertEqual(restored.inbox(1),c.inbox(1))
        c.step([0]*4,{2:'reply'});restored.step([0]*4,{2:'reply'})
        self.assertEqual(c.world.frame(),restored.world.frame())
        for i in range(4):self.assertEqual(c.inbox(i),restored.inbox(i))

if __name__=='__main__':unittest.main()
