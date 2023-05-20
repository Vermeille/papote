from papote.model import make_transformer, list_models

if __name__ == '__main__':
    for size in list_models().keys():
        print(size)
        model = make_transformer(size, 4096, 512)
        print('   ', model.num_parameters() / 1e6, 'M params')
        print('   ',
              model.num_parameters_without_embeddings() / 1e6,
              'M params without embeddings')
        print()
