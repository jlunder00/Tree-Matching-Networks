# Authored by: Jason Lunder, Github: https://github.com/jlunder00/

def get_sweep_config():
    """Generate wandb sweep configuration"""
    return {
        'method': 'bayes',
        'metric': {
            'name': 'val.loss',
            'goal': 'minimize'
        },
        'parameters': {
            # Model dimensions
            'model.node_state_dim': {
                'values': [256, 512, 768, 1024]
            },
            'model.edge_state_dim': {
                'values': [64, 128, 192, 256]
            },
            'model.node_hidden_sizes': {
                'values': [
                    [256], 
                    [512], 
                    [768]
                ]
            },
            'model.edge_hidden_sizes': {
                'values': [
                    [128], 
                    [256], 
                    [384]
                ]
            },
            'model.graph_rep_dim': {
                'values': [512, 768, 1024, 1792]
            },
            'model.graph_transform_sizes': {
                'values': [
                    [256],
                    [512],
                    [768]
                ]
            },
            
            # Training parameters
            'model.n_prop_layers': {
                'values': [3, 5, 7]
            },
            'model.temperature': {
                'min': 0.01,
                'max': 0.1
            },
            'train.learning_rate': {
                'distribution': 'log_uniform_values',
                'min': 1e-6,
                'max': 1e-4
            },
            'model.use_reverse_direction': {
                'values': [True, False]
            },
            'model.reverse_dir_param_different': {
                'values': [True, False]
            }
        }
    }
