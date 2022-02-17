#%%
from brian2 import *
from brian2tools import *
from matplotlib import pyplot as plt


start_scope()

eqs = '''
dv/dt = (I-v)/tau : 1
I : 1
tau : second
'''

G = NeuronGroup(3, eqs, threshold='v>1', reset='v=0', method='exact')
G.I = [2, 0, 0]
G.tau = [10, 100, 100] * ms

# Comment these two lines out to see what happens without Synapses
S = Synapses(G, G, 'w : 1', on_pre='v_post += w')
S.connect(i=0, j=[1, 2])
S.w = 'j*0.2'
S.delay = 'j*2*ms'

M = StateMonitor(G, 'v', record=True)

run(50*ms)

plot(M.t/ms, M.v[0], label='Neuron 0')
plot(M.t/ms, M.v[1], label='Neuron 1')
plot(M.t/ms, M.v[2], label='Neuron 2')
xlabel('Time (ms)')
ylabel('v')
legend()
plt.show()

#%%

start_scope()

N = 10
G = NeuronGroup(N, 'v:1')
S = Synapses(G, G)
S.connect(condition='i!=j', p=0.2)

def visualize_connectivity(S):
    Ns = len(S.source)
    Nt = len(S.target)
    figure(figsize=(10, 4))
    subplot(1,2,1)
    plot(zeros(Ns), arange(Ns), 'ok', ms=10)
    plot(ones(Nt), arange(Nt), 'ok', ms=10)

    for i, j in zip(S.i, S.j):
        plot([0, 1], [i, j], '-k')
    xticks([0, 1], ['Source', 'Target'])
    ylabel('Neuron index')
    xlim(-0.1, 1.1)
    ylim(-1, max(Ns, Nt))
    subplot(122)
    plot(S.i, S.j, 'ok')
    xlim(-1, Ns)
    ylim(-1, Nt)
    xlabel('Source neuron index')
    ylabel('Target neuron index')

visualize_connectivity(S)


#%%

start_scope()

taupre = 20*ms
taupost = 20*ms
wmax = 0.01
Apre = 0.01
Apost = -Apre * taupre/taupost * 1.05

G = NeuronGroup(2, 'v:1', threshold='t>(1+i)*10*ms', refractory=100*ms)

S = Synapses(G, G,
            '''
            w:1
            dapre/dt = -apre/taupre : 1 (clock-driven)
            dapost/dt = -apost/taupost : 1 (clock-driven)
            ''',
            on_pre='''
            v_post += w
            apre +=Apre
            w = clip(w+apost, 0, wmax)
            ''',
            on_post='''
            apost += Apost
            w = clip(w+apre, 0, wmax)
            ''')

S.connect(i=0, j=1)
M = StateMonitor(S, ['w', 'apre', 'apost', 'v'], record=True)

run(400*ms)

figure(figsize=(6, 12))
subplot(311)
plot(M.t/ms, M.apre[0], label='apre')
plot(M.t/ms, M.apost[0], label='apost')
xlabel('Time (ms)')
legend()


subplot(312)
plot(M.t/ms, M.w[0], label='w')
legend(loc='best')
xlabel('Time (ms)')

subplot(313)
plot(M.t/ms, M.v[0], label='v')
legend(loc='best')
xlabel('Time (ms)')

#%%

start_scope()

taupre = taupost = 20*ms
Apre = 0.01
Apost = -Apre * taupre/taupost * 1.05
tmax = 50*ms
N = 100

# Presynaptic neuron G spike at times from 0 to tmax
# Postsynaptic neurons G spike at times from tmax to 0
# So difference in spike times will vary from -tmax to +tmax
G = NeuronGroup(N, 'tspike:second', threshold='t>tspike', refractory=100*ms)
H = NeuronGroup(N, 'tspike:second', threshold='t>tspike', refractory=100*ms)
G.tspike = 'i * tmax/(N-1)'
H.tspike = '(N-1-i) * tmax/(N-1)'

S = Synapses(G, H,
            '''
            w:1
            dapre/dt = -apre/taupre : 1 (clock-driven)
            dapost/dt = -apost/taupost : 1 (clock-driven)
            ''',
            on_pre='''
            apre +=Apre
            w = w+apost
            ''',
            on_post='''
            apost += Apost
            w = w+apre
            ''')
S.connect(j='i')
M = StateMonitor(S, ['w', 'apre', 'apost'], record=True)

run(tmax+1 * ms)

figure(figsize=(6, 12))

subplot(311)
plot((H.tspike-G.tspike)/ms, S.w)
xlabel(r'$\Delta t$ (ms)')
ylabel(r'$\Delta w$')
axhline(0, ls='-', c='k')

subplot(312)
plot(M.t/ms, M.apre[0], label='apre')
plot(M.t/ms, M.apost[0], label='apost')
xlim(0, 100)
xlabel('Time (ms)')
legend()
