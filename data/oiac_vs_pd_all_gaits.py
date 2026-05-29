import numpy as np
import matplotlib.pyplot as plt

base_size = 20
plt.rcParams.update({
    'font.size': base_size,        # Default text size
    'axes.titlesize': base_size + 5,   # Title size
    'axes.labelsize': base_size + 4,   # X and Y label size
    'xtick.labelsize': base_size + 2,  # X tick size
    'ytick.labelsize': base_size + 2,  # Y tick size
    'legend.fontsize': base_size + 2   # Legend size
})

# OIAC best
oiac_walk = [0.5537270811310405, 0.5533163455178414, 0.5506688181161296, 0.5524524633625738,
             0.5514864322083259, 0.5512504659189404, 0.5504016618426806, 0.5511097161567668,
             0.5480342902224908, 0.5520875084287874]

oiac_amble = [0.32157784926404637, 0.32423919515099897, 0.3243163832088597, 0.3259318894820075,
              0.3238458596979593, 0.32662368362994726, 0.32436493058631954, 0.3259889987916545,
              0.3258639197506331, 0.3254407770645309]

oiac_trot = [0.15846274232592766, 0.15822651491983594, 0.15875460466087538, 0.16037748958391682,
             0.1590493887115292, 0.15944344907400768, 0.1600992854200346, 0.16029615107952777,
             0.16044298542713234]

# PD best
pd_walk = [0.6434683054671155, 0.6342456393486017, 0.6324558446031082, 0.6318837371299442,
           0.6338236370945655, 0.6319784827814005, 0.6324081400061425, 0.6337724631004068,
           0.6326403394475593, 0.6322633646335148]

pd_amble = [11.293929399200099, 1.3887936705568587, 3.613360391073241, 3.193430558615431,
            6.9730819244997395, 0.7979572274274593, 9.285332152509463, 0.8438447216649998,
            6.766314232361178, 4.54158231403756]

pd_amble = [min(1.0, x) for x in pd_amble] 

pd_trot = [0.24384973845456864, 0.23398779821581944, 0.23532548889746613, 0.23415857830848402,
           0.23537937004166795, 0.23400991270477148, 0.23534686459194043, 0.23434262626915264,
           0.23506159213656308, 0.23394275177827842]

gaits = ['Walk', 'Trot', 'Amble']
pd_means = [np.mean(pd_walk), np.mean(pd_trot), np.mean(pd_amble)]
pd_stds = [np.std(pd_walk), np.std(pd_trot), np.std(pd_amble)]
oiac_means = [np.mean(oiac_walk), np.mean(oiac_trot), np.mean(oiac_amble)]
oiac_stds = [np.std(oiac_walk), np.std(oiac_trot), np.std(oiac_amble)]

x = np.arange(len(gaits))
width = 0.30

fig, ax = plt.subplots(figsize=(9, 5))
bars_pd = ax.bar(x - width/2, pd_means, width, yerr=pd_stds, capsize=5,
                 color='#4a90d9', label='PD', edgecolor='black', linewidth=0.6)
bars_oiac = ax.bar(x + width/2, oiac_means, width, yerr=oiac_stds, capsize=5,
                   color='#3ba33b', label='OIAC', edgecolor='black', linewidth=0.6)

ax.set_ylabel('Cost of Transport (CoT)')
ax.set_title('PD vs OIAC CoT Comparison Across Gaits')
ax.set_xticks(x)
ax.set_xticklabels(gaits)
ax.legend()
ax.grid(axis='y', alpha=0.3)

for bar in list(bars_pd) + list(bars_oiac):
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.02,
            f'{h:.2f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('cot_comparison.png', dpi=150)
plt.show()