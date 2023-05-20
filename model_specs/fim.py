def list_models():
    return {
        'fim-xxs': {
            'hidden_size': 384,
            'num_heads': 6,
            'head_size': 64,
            'num_layers': 6
        },
        'fim-xs': {
            'hidden_size': 512,
            'num_heads': 8,
            'head_size': 64,
            'num_layers': 8
        },
        'fim-s': {
            'hidden_size': 768,
            'num_heads': 12,
            'head_size': 64,
            'num_layers': 12
        },
        'fim-m': {
            'hidden_size': 1024,
            'num_heads': 16,
            'head_size': 64,
            'num_layers': 24
        },
        'fim-l': {
            'hidden_size': 1536,
            'num_heads': 16,
            'head_size': 96,
            'num_layers': 24
        },
        'fim-xl': {
            'hidden_size': 2048,
            'num_heads': 16,
            'head_size': 128,
            'num_layers': 24
        },
        'fim-xxl': {
            'hidden_size': 2560,
            'num_heads': 32,
            'head_size': 80,
            'num_layers': 32
        },
        'fim-xxxl': {
            'hidden_size': 4096,
            'num_heads': 32,
            'head_size': 128,
            'num_layers': 32
        }
    }
