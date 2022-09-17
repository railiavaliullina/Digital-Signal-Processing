import matplotlib.pyplot as plt


def plot_all_hists_for_signal(signal, quantized_signal, error, signal_name=''):
    fig, axs = plt.subplots(3, 1, figsize=(10, 10))
    axs = axs.flatten()
    hists_args = [signal, quantized_signal, error]
    hists_names = ['_signal_to_quantize', '_quantized_signal', '_signal_error']
    plt.title(signal_name)
    fig.tight_layout(pad=4.5)
    for i, ax in enumerate(axs):
        ax.hist(hists_args[i], bins=len(hists_args[i]))
        ax.set_title(signal_name + hists_names[i], fontsize=10, fontweight="bold")
    plt.show()
    # plt.savefig(f'hists/{signal_name}_hists.png')

