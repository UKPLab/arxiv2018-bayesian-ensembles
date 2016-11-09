'''
Created on Oct 19, 2016

@author: Melvin Laux
'''
import synth_data
import matplotlib.pyplot as plt

synth_data.read_config_file('config.ini')

p = []
r = []
f1 = []
s = []

for acc in xrange(5):
    synth_data.accuracy = acc
    synth_data.build_crowd_model()
    synth_data.simulate_ground_truth()
    synth_data.simulate_crowd_annotators()
    synth_data.majority_voting()
    synth_data.save_generated_data('data'+str(acc)+'.csv')
    p_, r_, f1_, s_ = synth_data.calculate_metrics()
    p.append(p_)
    r.append(r_)
    f1.append(f1_)
    s.append(s_)
    
plt.clf()
plt.plot(range(5),p,label='Line')
plt.show()

if __name__ == '__main__':
    pass