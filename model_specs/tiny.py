def list_models():
    return {
        'tiny-1M': {
            'hidden_size': 64,
            'num_heads': 16,
            'head_size': 4,
            'num_layers': 8
        },
        'tiny-3M': {
            'hidden_size': 128,
            'num_heads': 16,
            'head_size': 8,
            'num_layers': 8
        },
        'tiny-8M': {
            'hidden_size': 256,
            'num_heads': 16,
            'head_size': 16,
            'num_layers': 8
        },
        'tiny-28M': {
            'hidden_size': 512,
            'num_heads': 16,
            'head_size': 32,
            'num_layers': 8
        },
        'tiny-33M': {
            'hidden_size': 768,
            'num_heads': 16,
            'head_size': 48,
            'num_layers': 4
        }
    }
