import numpy as np
import os
import matplotlib.pyplot as plt
import time

def save_best_model(model_history, acc, folder='tmp', file_name='model', loss_name='loss', acc_name='acc', tmp_name='tmp/tmp_weights.hdf5'):
    # model.save('tmp/model_' + file_name + '_' + str(logs_to_save['val_acc'][-1]) + '.hdf5')
    best_model_name = folder + '/model_' + file_name + '_acc_' + str(round(acc, 5)) + '.hdf5'
    os.system('mv ' + tmp_name + ' ' + best_model_name)
    print('Save model file to', best_model_name)


def save_logs_models(model, model_history0, acc, folder='tmp', lr_hist=None, file_name='model', loss_name='loss', acc_name='acc', tmp_name='tmp/tmp_weights.hdf5'):
    model_history = model_history0.history.copy()
    if lr_hist:
        model_history['lr'] = lr_hist

    best_log_name = folder + '/log_' + file_name + '_acc_' + str(round(acc, 5)) + '.txt'

    with open(best_log_name,'w') as f:
        header = sorted(model_history.keys())
        f.write(",".join(header))

        for i in range(len(model_history[header[0]])):
            f.write('\n')
            for j in range(len(header)-1):
                f.write(str(model_history[header[j]][i])+',')
            f.write(str(model_history[header[-1]][i]))

    print('Save log file to', best_log_name)
    save_best_model(model_history, acc, folder=folder,file_name=file_name, loss_name=loss_name, acc_name=acc_name, tmp_name=tmp_name)
    return best_log_name


def plot_log(model_history, lr_hist, file_name='model', save_fig = False, loss_name='loss', acc_name='acc'):
    plt.figure(figsize=(15,10))
    plt.subplot(221)
    plt.semilogy(model_history.history[loss_name], 'b')
    plt.semilogy(model_history.history['val_'+loss_name], 'r')
    plt.legend(['train', 'val'])
    plt.xlabel('epoch')
    plt.ylabel('Loss')
    plt.grid()

    plt.subplot(222)
    plt.semilogy(model_history.history[acc_name], 'b')
    plt.semilogy(model_history.history['val_'+acc_name], 'r')
    plt.legend(['train', 'val'])
    plt.xlabel('epoch')
    plt.ylabel('Accuracy')
    plt.grid()
    tt = 'Val accuracy = ' + str(max(model_history.history['val_'+acc_name])) + ' @ epoch '
    tt += str(np.argmax(np.array(model_history.history['val_'+acc_name])))
    plt.title(tt)

    plt.subplot(223)
    plt.semilogy(lr_hist, 'b')
    plt.legend(['Learning Rate'])
    plt.xlabel('epoch')
    plt.ylabel('Learning rate')
    plt.grid()

    if save_fig:
        fig_name = 'tmp/fig_' + file_name + '_' + str(max(model_history.history['val_'+acc_name])) +'.png'
        plt.savefig(fig_name)
        print('Save fig to:', fig_name)
    plt.show()


def GPU_config(device='0,1', usage=0.45, allow_growth=True):
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    import subprocess

    if not device:
        # visible_device="" for CPU only
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        print('Allocate to CPU only')
    else:
        # Dynamic allocation
        device_list = device.replace(' ', '').split(',')
        dev_cont = len(device_list)
        utilization = []
        # pre = 'nvidia-smi --query-gpu=utilization.gpu,utilization.memory --format=csv,noheader,nounits --id='

        ## Current usage
        # pre = 'nvidia-smi --query-gpu=utilization.memory --format=csv,noheader,nounits --id='

        # Allocated usage
        pre = 'nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits --id='
        for d in device_list:
            cmd = pre+d
            utilization.append(int(subprocess.check_output(cmd.split(' '))[:-1])) # remove '\n'
        sel_dev = np.argmin(utilization)
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = usage * (1 - min(utilization)/100)
        config.gpu_options.visible_device_list = str(device_list[sel_dev])
        print('Allocate to device:', device_list[sel_dev])
        config.gpu_options.allow_growth = allow_growth
        set_session(tf.Session(config=config))


class record():
    def __init__(self):
        self.name = []
        self.header = []
        self.data = []


    def read_header(self, log_file):
        with open(log_file, 'r') as f:
            header = f.readline().split(',')
            header[-1] = header[-1][:-1] # Remove '\n'
        self.name.append(log_file)
        self.header.append(header)


    def read_log(self, log_file):
        self.read_header(log_file)
        tmp = np.loadtxt(log_file, delimiter=',', skiprows = 1)
        if len(tmp.shape) == 1:
            tmp = np.expand_dims(tmp, axis=0)
        self.data.append(tmp)


    def read_folder(self, path):
        files = os.listdir(path)
        log_files = []
        for i in files:
            if len(i) > 5 :
                if i[:3] == 'log' and i[-3:] == 'txt':
                    log_files.append(path + '/' + i)
        log_files = sorted(log_files)
        if self.skip >= 0:
            log_files = log_files[self.skip:]
        else:
            log_files = log_files[:self.skip]

        if not log_files:
            print('No log file (.txt) in the folder:', path)
        else:
            print('Log files:')
            [print(f) for f in log_files]
            print('In total: %d log files\n'%(len(log_files)))

        [self.read_log(fi) for fi in log_files]


    def plot_log(self, loss_name='loss', acc_name='acc'):
        plt.figure(figsize=(15,10))

        plt.subplot(221)
        for i, d in enumerate(self.data):
            loss_name_tmp = loss_name
            if loss_name not in self.header[i]:
                for h in self.header[i]:
                    if 'loss' in h:
                        loss_name_tmp = h
                        print('%d: use %s\n' % (i, loss_name_tmp))
                        break
            if len(self.name) == len(self.labels):
                plt.plot(d[:, self.header[i].index(loss_name_tmp)], label='train_loss_'+self.labels[i])
                plt.plot(d[:, self.header[i].index('val_'+loss_name_tmp)], label='val_loss_'+self.labels[i])
            else:
                plt.plot(d[:, self.header[i].index(loss_name_tmp)], label='train_loss_'+str(i))
                plt.plot(d[:, self.header[i].index('val_'+loss_name_tmp)], label='val_loss_'+str(i))
        plt.legend(loc=0)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid()

        plt.subplot(222)
        for i, d in enumerate(self.data):
            acc_name_tmp = acc_name
            if acc_name not in self.header[i]:
                for h in self.header[i]:
                    if 'acc' in h:
                        acc_name_tmp = h
                        print('%d: use %s' % (i, acc_name_tmp))
                        break
            if len(self.name) == len(self.labels):
                plt.plot(d[:, self.header[i].index(acc_name_tmp)], label='train_acc_'+self.labels[i])
                plt.plot(d[:, self.header[i].index('val_'+acc_name_tmp)], label='val_acc_'+self.labels[i])
            else:
                plt.plot(d[:, self.header[i].index(acc_name_tmp)], label='train_acc_'+str(i))
                plt.plot(d[:, self.header[i].index('val_'+acc_name_tmp)], label='val_acc_'+str(i))
        plt.legend(loc=0)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.grid()

        if len(self.data) == 1:
            tt = 'Val accuracy = %.3f @ epoch %d' % (max(self.data[0][:, self.header[0].index('val_'+acc_name_tmp)]),
                np.argmax(self.data[0][:, self.header[0].index('val_'+acc_name_tmp)]))
            plt.title(tt)

        plt.subplot(223)
        for i, d in enumerate(self.data):
            if 'lr' in self.header[i]:
                if len(self.name) == len(self.labels):
                    plt.plot(d[:, self.header[i].index('lr')], label=self.labels[i])
                else:
                    plt.plot(d[:, self.header[i].index('lr')], label=str(i))
        plt.legend(loc=0)
        plt.xlabel('epoch')
        plt.ylabel('Learning Rate')
        plt.grid()

        if self.savefig:
            p = './plot.png'
            plt.savefig(p)
            print('Save fig to', p)

        print('Logs:')
        for i in range(len(self.name)):
            print(str(i), ': ', self.name[i])


    def norm_plot(self, loss_name='loss', acc_name='acc'):
        self.norm = [float(i) for i in self.norm]
        x = [np.linspace(0, self.data[i].shape[0], self.data[i].shape[0], endpoint=False)*self.norm[i] for i in range(len(self.data))]

        plt.figure(figsize=(15,7))
        plt.subplot(121)
        for i, d in enumerate(self.data):
            loss_name_tmp = loss_name
            if loss_name not in self.header[i]:
                for h in self.header[i]:
                    if 'loss' in h:
                        loss_name_tmp = h
                        print('%d: use %s\n' % (i, loss_name_tmp))
                        break
            if len(self.name) == len(self.labels):
                plt.semilogy(x[i], d[:, self.header[i].index(loss_name_tmp)], label='train_loss_'+self.labels[i])
                plt.semilogy(x[i], d[:, self.header[i].index('val_'+loss_name_tmp)], label='val_loss_'+self.labels[i])
            else:
                plt.semilogy(x[i], d[:, self.header[i].index(loss_name_tmp)], label='train_loss_'+str(i))
                plt.semilogy(x[i], d[:, self.header[i].index('val_'+loss_name_tmp)], label='val_loss_'+str(i))
        plt.legend(loc=0)
        plt.xlabel('Training time (h)')
        plt.ylabel('Loss')
        plt.grid()

        plt.subplot(122)
        for i, d in enumerate(self.data):
            acc_name_tmp = acc_name
            if acc_name not in self.header[i]:
                for h in self.header[i]:
                    if 'acc' in h:
                        acc_name_tmp = h
                        print('%d: use %s' % (i, acc_name_tmp))
                        break
            if len(self.name) == len(self.labels):
                plt.plot(x[i], d[:, self.header[i].index(acc_name_tmp)], label='train_acc_'+self.labels[i])
                plt.plot(x[i], d[:, self.header[i].index('val_'+acc_name_tmp)], label='val_acc_'+self.labels[i])
            else:
                plt.plot(x[i], d[:, self.header[i].index(acc_name_tmp)], label='train_acc_'+str(i))
                plt.plot(x[i], d[:, self.header[i].index('val_'+acc_name_tmp)], label='val_acc_'+str(i))
        plt.legend(loc=0)
        plt.xlabel('Training time (h)')
        plt.ylabel('Accuracy')
        plt.grid()


    def main(self, inputs, skip=0, merge=False, loss_name='loss', acc_name='acc', savefig=False, labels=[], norm=[]):
        self.skip = skip
        self.merge = merge
        self.labels = labels
        self.savefig = savefig
        self.norm = norm

        if not inputs: # inputs is empty, path_all is empty
            inputs = ['tmp']

        for name_i in inputs:
            if os.path.isdir(name_i):
                try:
                    self.read_folder(name_i)
                except:
                    print ('No such folder! - ', name_i)
            else:
                if '.txt' in name_i:
                    try:
                        self.read_log(name_i)
                    except:
                        print ('No such file - ' + name_i)
                elif '.hdf5' in name_i:
                    ind = name_i.index('model')
                    if ind == -1:
                        print('No \'model\' in model file name')
                        exit()
                    tmp = name_i[:ind]+'log'+name_i[ind+5:-4]+'txt'
                    try:
                        print ('Try \'.txt\' instead of \'.hdf5\': ', tmp)
                        self.read_log(tmp)
                    except:
                        print ('No such file - ' + tmp)
                else:
                    print ('Wrong file type - ' + name_i)
        if self.name:
            if self.merge:
                for h in self.header:
                    if h != self.header[0]:
                        print('Different headers')
                        exit()
                print('Merging to one log file...')
                self.data = [d for _,d in sorted(zip(self.name, self.data), key=lambda pair: pair[0])]
                self.name = sorted(self.name)
                data = self.data[0]
                for i in range(1, len(self.data)):
                    data = np.concatenate((data, self.data[i]), axis=0)
                self.data = [data]
            self.plot_log(loss_name=loss_name, acc_name=acc_name)
            if len(self.name) == len(self.norm):
                self.norm_plot(loss_name=loss_name, acc_name=acc_name)
            plt.show()



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(usage='python utils.py [-f FILE]',
        description='Plot training curve from the log folder or files (default /tmp)',
        epilog='Written by Leo Wang')
    parser.add_argument("-f", "--file", help="model files or folders, default is 'tmp'",
                        type=str, default='', nargs='+')
    parser.add_argument("--save", action="store_true",
                    help="save figure", default=False)
    parser.add_argument("--loss", help="loss name, default is 'loss'",
                        type=str, default='loss')
    parser.add_argument("-a", "--accuracy", help="accuracy name, default is 'acc'",
                        type=str, default='acc')
    parser.add_argument("-s", "--skip", help="skip first/last few log files",
                        type=int, default=0)
    parser.add_argument("-m", "--merge", help="merge log files in the folder",
                        action="store_true", default=False)
    parser.add_argument("-l", "--labels", help="labels for plotting",
                        type=str, default='', nargs='+')
    parser.add_argument("-n", "--norm_plot", help="Normalized plot of training time for comparison, t is the training time for each epoch",
                        type=str, default='', nargs='+')
    args = parser.parse_args()

    record = record()
    record.main(inputs=args.file, skip=args.skip, merge=args.merge, loss_name=args.loss, acc_name=args.accuracy, savefig=args.save, labels=args.labels, norm=args.norm_plot)
