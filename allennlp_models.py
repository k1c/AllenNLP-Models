"""
Usage for Semantic Role Labeling:
python3 allennlp_models.py \
    https://s3-us-west-2.amazonaws.com/allennlp/models/srl-model-2018.02.27.tar.gz \
    biased_sentences.json --output-file biased_sentences_srl.json

Usage for Co-reference Resolution:
python3 allennlp_models.py \
    https://s3-us-west-2.amazonaws.com/allennlp/models/coref-model-2018.02.05.tar.gz \
    biased_sentences.json --output-file biased_sentences_coref.json
"""

from allennlp.predictors import Predictor
from allennlp.models.archival import load_archive
from contextlib import ExitStack
import argparse
import json

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('archive_file', type=str, help='the archived model to make predictions with')
    parser.add_argument('input_file', type=argparse.FileType('r'), help='path to input file')

    parser.add_argument('--output-file', type=argparse.FileType('w'), help='path to output file')

    parser.add_argument('--batch-size', type=int, default=1, help='The batch size to use for processing')

    parser.add_argument('--cuda-device', type=int, default=-1, help='id of GPU to use (if any)')

    args = parser.parse_args()

    return args

def get_predictor(args):
    archive = load_archive(args.archive_file,
                           weights_file=None,
                           cuda_device=args.cuda_device,
                           overrides="")

    model_type = archive.config.get("model").get("type")
    if model_type != 'srl' and model_type != 'coref':
        raise Exception('the given model must be srl or coref.')

    if model_type == 'srl':
        return Predictor.from_archive(archive, 'semantic-role-labeling'),model_type;
    if model_type == 'coref':
        return Predictor.from_archive(archive, 'coreference-resolution'),model_type;


def run(predictor,
        model_type,
        input_file,
        output_file,
        batch_size,
        print_to_console,
        cuda_device):

    def _run_predictor(batch_data):
        if len(batch_data) == 1:
            #result = predictor.predict_json(batch_data[0], cuda_device)
            result = predictor.predict_json(batch_data[0])

            # Batch results return a list of json objects, so in
            # order to iterate over the result below we wrap this in a list.
            results = [result]
        else:
            results = predictor.predict_batch_json(batch_data, cuda_device)

        for model_input, output in zip(batch_data, results):
            string_output = predictor.dump_line(output)
            if print_to_console:
                print("input: ", model_input)
                print("prediction: ", string_output)
            if output_file:
                output_file.write(string_output)

    batch_data = []
    for line in input_file:
        if not line.isspace():
            if model_type == 'srl':
                line = {"sentence":line.strip()}
            elif model_type == 'coref':
                line = {"document":line.strip()}
            line = json.dumps(line)
            json_data = predictor.load_line(line)
            batch_data.append(json_data)
            if len(batch_data) == batch_size:
                _run_predictor(batch_data)
                batch_data = []

    if batch_data:
        _run_predictor(batch_data)


def main():
    args = get_arguments()
    predictor,model_type = get_predictor(args)
    output_file = None
    print_to_console = False

    with ExitStack() as stack:
        input_file = stack.enter_context(args.input_file)
        if args.output_file:
            output_file = stack.enter_context(args.output_file)

        if not args.output_file:
            print_to_console = True

        run(predictor,
            model_type,
            input_file,
            output_file,
            args.batch_size,
            print_to_console,
            args.cuda_device)

if __name__ == '__main__':
    main()
