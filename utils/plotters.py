import matplotlib.pyplot as plt
import numpy as np

def plot(orig,inferred,jetid=0,variableid=0,process="SM",colormap='Reds',nbins=100):
    varidtostr = {0:['$p_\\mathrm{T}$ (GeV)','px','pt'],1:['$\eta$','py','eta'],2:['$\phi$','pz','phi']}
    processtolabel = {'wp':'$M_{W\'}=3\,\\mathrm{TeV}$, $M_{B\'}=400\,\\mathrm{GeV}$',
                      'SM':'SM','wkk':'$M_{W_{KK}}=2\,\\mathrm{TeV}$, $M_R=400\,\\mathrm{GeV}$',
                      'qstar':'$M_{Q^{*}}=2\,\\mathrm{TeV}$, $M_W=400\,\\mathrm{GeV}$',
                    'xyy':'$M_{X=3\,\\mathrm{TeV}$, $M_Y=400\,\\mathrm{GeV}, $M_{Y\prime}=170\,\\mathrm{GeV}$'
                     }
    fig, axs = plt.subplots(nrows=2, ncols=5, figsize=(16, 7))
    nparts = [i for i in range(10)]
    
    for ax,nth_part in zip(axs.flat,nparts):
        #print(nth_part)

        mean = np.mean(orig[jetid,:,nth_part,variableid])
        std = np.std(orig[jetid,:,nth_part,variableid])
        mean_model = np.mean(inferred[jetid,:,nth_part,variableid])
        std_model = np.std(inferred[jetid,:,nth_part,variableid])

        modelbins = np.linspace(mean_model-3*std,mean_model+3*std,nbins)
        #print(np.max(inferred[0,:,nth_part,0]))
        origbins = np.linspace(mean-3*std,mean+3*std,nbins)
        z, xedges, yedges = np.histogram2d(inferred[jetid,:,nth_part,variableid], orig[jetid,:,nth_part,variableid], bins=(origbins, origbins))
        ax.tick_params(direction="in")
        ax.imshow(z, interpolation='nearest', origin='lower',
             extent=[yedges[0], yedges[-1], xedges[0], xedges[-1]],  cmap=plt.get_cmap(colormap))
        denote_nth = 'th'
        if str(nth_part)[-1] == '0':
            denote_nth = 'st'
        elif str(nth_part)[-1] == '1':
            denote_nth = 'nd'
        elif str(nth_part)[-1] == '2':
            denote_nth = 'rd'
        plt.text(0.06,0.88,"$%s^\\mathrm{%s}$ const."%(nth_part+1,denote_nth),fontsize=11,transform = ax.transAxes)
        plt.text(0.06,0.82,"%.3f"%np.corrcoef(inferred[jetid,:,nth_part,variableid], orig[jetid,:,nth_part,variableid])[0,1],fontsize=11,transform = ax.transAxes)
        if nth_part == 5:
            ax.set_xlabel("orig jet %s"%varidtostr[variableid][0],fontsize=10)
            ax.set_ylabel("NF-reconstructed orig jet %s"%varidtostr[variableid][0],fontsize=10)

    fig.suptitle("%s $J_%i$"%(processtolabel[process],jetid+1),fontsize=20)

    #fig.tight_layout()

    _ = plt.show()
    print("-------------------------------------------------")
    #plt.savefig("2D_%s_j%i_%s.pdf"%(process,jetid+1,varidtostr[variableid][1]),bbox_inches='tight')
    fig.savefig("2D_%s_j%i_%s.png"%(process,jetid+1,varidtostr[variableid][2]),dpi=450,bbox_inches='tight')