from piq import fsim


def compute_all_metrics(y, y_hat):
    fsim_loss = fsim(y, y_hat, chromatic=False)
    return dict(
        fsim=fsim_loss,
    )
