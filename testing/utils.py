import models
import tasks
from runner import Runner

def run_electrode_test(cfg):
#run a test for a single electrode
    test_results = []
    orig_cfg = cfg.copy()
    for i in range(cfg.test.test_runs):
        cfg = orig_cfg.copy()#data.cfg.cached_transcript_aligns gets modified downstream
        if i>0:
            cfg.data.reload_caches=False #don't need to cache after first run
        task = tasks.setup_task(cfg.task)
        task.load_datasets(cfg.data, cfg.preprocessor)
        model = task.build_model(cfg.model)
        criterion = task.build_criterion(cfg.criterion)
        runner = Runner(cfg.exp.runner, task, model, criterion)
        best_model = runner.train()
        test_results.append(runner.test(best_model))
    return test_results

def run_subject_test(data_cfg, brain_runs, electrodes, cfg):
    cache_path = None
    if "cache_input_features" in data_cfg:
        cache_path = data_cfg.cache_input_features

    subject_test_results = {}
    for e in electrodes:
        data_cfg.electrodes = [e]
        data_cfg.brain_runs = brain_runs
        if cache_path is not None:
            #cache_path needs to identify the pretrained model
            e_cache_path = os.path.join(cache_path, data_cfg.subject, data_cfg.name ,e)
            log.info(f"logging input features in {e_cache_path}")
            data_cfg.cache_input_features = e_cache_path
        cfg.data = data_cfg
        test_results = run_electrode_test(cfg)
        subject_test_results[e] = test_results
    return subject_test_results
