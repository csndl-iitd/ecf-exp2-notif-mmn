# first line: 1
@memory.cache
def run_autoreject(epochs, n_jobs=1, cv=3):
    # First, try a simple threshold-based rejection
    reject = get_rejection_threshold(epochs, cv=cv, random_state=42)
    epochs_clean = epochs.copy().drop_bad(reject=reject)
    
    # If more than 20% of epochs were rejected, use AutoReject
    if (len(epochs) - len(epochs_clean)) / len(epochs) > 0.1:
        print("Simple threshold rejected too many epochs. Using AutoReject...")
        ar = AutoReject(n_jobs=n_jobs, random_state=42, verbose=True)
        ar.fit(epochs)
        epochs_clean = ar.transform(epochs)
    
    return epochs_clean
