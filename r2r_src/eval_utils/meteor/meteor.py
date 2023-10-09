#!/usr/bin/env python

# Python wrapper for METEOR implementation, by Xinlei Chen
# Acknowledge Michael Denkowski for the generous discussion and help 

import os
import sys
import subprocess
import threading

# Assumes meteor-1.5.jar is in the same directory as meteor.py.  Change as needed.
METEOR_JAR = os.path.join(os.path.dirname(__file__), './meteor-1.5.jar')


# print METEOR_JAR

class Meteor:

    def __init__(self):
        self.env = os.environ
        self.env['LC_ALL'] = 'en_US.UTF_8'
        self.meteor_cmd = ['java', '-jar', '-Xmx2G', METEOR_JAR,
                           '-', '-', '-stdio', '-l', 'en', '-norm']
        self.meteor_p = subprocess.Popen(self.meteor_cmd,
                                         cwd=os.path.dirname(os.path.abspath(__file__)),
                                         stdin=subprocess.PIPE,
                                         stdout=subprocess.PIPE,
                                         stderr=subprocess.PIPE,
                env=self.env, universal_newlines=True, bufsize=1)
        # Used to guarantee thread safety
        self.lock = threading.Lock()

    def compute_score(self, gts, res, gt_paths=None):
        assert (gts.keys() == res.keys())
        imgIds = sorted(list(gts.keys()))
        scores = []

        eval_line = 'EVAL'
        self.lock.acquire()
        for i in imgIds:
            assert (len(res[i]) == 1)
            stat = self._stat(res[i][0], gts[i])
            eval_line += ' ||| {}'.format(stat)

        # Send to METEOR
        self.meteor_p.stdin.write(eval_line + '\n')

        # Collect segment scores
        for i in range(len(imgIds)):
            score = float(self.meteor_p.stdout.readline().strip())
            scores.append(score)

        # Final score
        final_score = float(self.meteor_p.stdout.readline().strip())
        self.lock.release()

        return final_score, scores

    def method(self):
        return "METEOR"

    def _stat(self, hypothesis_str, reference_list):
        # SCORE ||| reference 1 words ||| reference n words ||| hypothesis words
        hypothesis_str = hypothesis_str.replace('|||', '').replace('  ', ' ')
        score_line = ' ||| '.join(('SCORE', ' ||| '.join(reference_list), hypothesis_str))
        self.meteor_p.stdin.write(score_line+'\n')
        return self.meteor_p.stdout.readline().strip()

    def _score(self, hypothesis_str, reference_list):
        self.lock.acquire()
        # SCORE ||| reference 1 words ||| reference n words ||| hypothesis words
        hypothesis_str = hypothesis_str.replace('|||', '').replace('  ', ' ')
        score_line = ' ||| '.join(('SCORE', ' ||| '.join(reference_list), hypothesis_str))
        w = '{}\n'.format(score_line)
        self.meteor_p.stdin.write(bytes(w, 'UTF-8'))
        stats = self.meteor_p.stdout.readline().strip()
        eval_line = 'EVAL ||| {}'.format(stats)
        # EVAL ||| stats 
        self.meteor_p.stdin.write(f'{eval_line}\n')
        self.meteor_p.stdout.readline()
        # bug fix: there are two values returned by the jar file, one average, and one all, so do it twice
        # thanks for Andrej for pointing this out
        score = float(self.meteor_p.stdout.readline().strip())
        self.lock.release()
        return score

    def __del__(self):
        self.lock.acquire()
        self.meteor_p.stdin.close()
        self.meteor_p.kill()
        self.meteor_p.wait()
        self.lock.release()


if __name__ == "__main__":

    assert os.path.isfile(METEOR_JAR), 'you must have meteor-1.5.jar! Check README.md instructions in eval/ folder.'

    jobid = sys.argv[1] if len(sys.argv) >= 2 else ''

    m = Meteor()
    import json

    records = [(
               'Go straight and exit the room then go up the stairs on your left until you get to the top and wait there by the statue.',
               [
                   'Go straight and exit the room then go up the stairs on your left until you get to the top and wait there by the statue.',
                   'Exit through the door to the right. Walk by the bathroom and through another door into the landing. Go all the way up the stairs. Wait there.'])]
    scores = []
    for r, c in records:
        score = m._score(r, c)
        scores.append(score)

    out = {'scores': scores, 'average_score': sum(scores) / len(scores)}
    json.dump(out, open(OUTPUT_FILE, 'w'))
