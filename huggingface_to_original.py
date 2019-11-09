from transformers import TFGPT2LMHeadModel, GPT2Config
import os
from shutil import copyfile
import sys
import argparse
import tensorflow as tf

# modified from https://gist.github.com/batzner/7c24802dd9c5e15870b4b56e22135c96
def rewriter(checkpoint_dir, dry_run):
    checkpoint = tf.compat.v1.train.get_checkpoint_state(checkpoint_dir)
    with tf.compat.v1.Session() as sess:
        for var_name, _ in tf.train.list_variables(checkpoint_dir):
            # Load the variable
            var = tf.train.load_variable(checkpoint_dir, var_name)

            if var_name == '_CHECKPOINTABLE_OBJECT_GRAPH':
                continue
            # Set the new name
            new_name = var_name
            #print(new_name)
            if new_name[:12] != 'transformer/':
                print('!!!!!!!!!!', var_name, var.shape)
                assert False
                continue
            new_name = new_name[12:].replace('/.ATTRIBUTES/VARIABLE_VALUE', '')
            new_name = new_name.replace('weight', 'w')
            new_name = new_name.replace('bias', 'b')
            new_name = new_name.replace('beta', 'b')
            new_name = new_name.replace('gamma', 'g')
            if 'wpe' in new_name:
                new_name = 'wpe'
            if 'wte' in new_name:
                new_name = 'wte'
            new_name = 'model/' + new_name
            new_name = new_name.replace('/h/', '/h')
            if 'ln' in new_name or '/b' in new_name:
                var = var.reshape((-1))
            if '/w' in new_name and not ('wpe' in new_name or 'wte' in new_name):
                var = var.reshape((1, *var.shape))

            if dry_run:
                print('%s would be renamed to %s.' % (var_name, new_name), var.shape)
            else:
                print('Renaming %s to %s.' % (var_name, new_name))
                # Rename the variable
                var = tf.compat.v1.Variable(var, name=new_name)

        if not dry_run:
            # Save the variables
            print('Saving...')
            saver = tf.compat.v1.train.Saver()
            sess.run(tf.compat.v1.global_variables_initializer())
            saver.save(sess, checkpoint_dir, write_meta_graph=False)
    tf.compat.v1.reset_default_graph()


parser = argparse.ArgumentParser(
    description='Convert Huggingface model to orignal OpenAI GPT2 format',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--in_model', metavar='MODEL', type=str, required=True, help='The model to convert from.')
parser.add_argument('--in_model_config', metavar='CONF', type=str, required=True, help='The config of model to convert from.')
parser.add_argument('--from_pytorch', default=False, action='store_true', help='Whether the model being converted from is a pytorch model.')
parser.add_argument('--out_path', metavar='PATH', type=str, required=True, help='The path to output to.')
parser.add_argument('--meta_from', metavar='META', type=str, required=True, help='Model from which to copy meta, hparams, vocab, etc.')

args = parser.parse_args()

model = TFGPT2LMHeadModel.from_pretrained(args.in_model, config=GPT2Config.from_pretrained(args.in_model_config), from_pt=args.from_pytorch)

os.makedirs(f'{args.out_path}', exist_ok=True)
model.save_weights(f'{args.out_path}/model.ckpt')
copyfile(f'{args.meta_from}/encoder.json', f'{args.out_path}/encoder.json')
copyfile(f'{args.meta_from}/hparams.json', f'{args.out_path}/hparams.json')
copyfile(f'{args.meta_from}/vocab.bpe', f'{args.out_path}/vocab.bpe')
copyfile(f'{args.meta_from}/model.ckpt.meta', f'{args.out_path}/model.ckpt.meta')

rewriter(f'{args.out_path}/model.ckpt', dry_run=False)