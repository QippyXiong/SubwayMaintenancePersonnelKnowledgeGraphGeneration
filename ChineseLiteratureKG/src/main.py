from controller import Controller

if __name__ == '__main__':
    controller = Controller()
    controller.load_ner('ner-bilstm-crf_v0.2')
    print( controller.ner_task('我爸是李马') )