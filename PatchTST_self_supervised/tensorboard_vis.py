import tensorboardX as tb

log_path = '/home/gayal/ssl-project/PatchTST/PatchTST_self_supervised/saved_models/hf_tuh/logs/events.out.tfevents.1743103684.cs-u-vivaldi'

# Load tensorboard log
log = tb.SummaryReader(log_path)
print(log)