import socket
from mpi4py import MPI
from botocore.exceptions import ClientError
import logging
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import numpy as np
import cv2
import boto3
import torchvision.models as models
from collections import defaultdict
import sys
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
num_of_slaves = 4

# for wave algorithm
sequence =[]
visited = []
#for terminate detection
children = []

"""
AWS connection to bucket, queue
"""
def send_sqs_message(message, url):
    sqs = boto3.client('sqs', region_name="eu-central-1", aws_access_key_id='AKIAILK5L3NMS5TLTKOQ',
                       aws_secret_access_key='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')
    response = sqs.send_message(
        QueueUrl=url,
        DelaySeconds=0,
        MessageAttributes={
            'Title': {
                'DataType': 'String',
                'StringValue': 'The Whistler'
            },
            'Author': {
                'DataType': 'String',
                'StringValue': 'John Grisham'
            },
            'WeeksOn': {
                'DataType': 'Number',
                'StringValue': '6'
            }
        },
        MessageBody=(
            message
        )
    )


def read_from_sqs(url):
    # Create SQS client
    sqs = boto3.client('sqs', region_name="eu-central-1", aws_access_key_id='AKIAILK5L3NMS5TLTKOQ',
                       aws_secret_access_key='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')

    queue_url = url

    # Receive message from SQS queue
    response = sqs.receive_message(
        QueueUrl=queue_url,
        AttributeNames=[
            'SentTimestamp'
        ],
        MaxNumberOfMessages=1,
        MessageAttributeNames=[
            'All'
        ],
        VisibilityTimeout=0,
        WaitTimeSeconds=0
    )

    message = response['Messages'][0]['Body']

    return message


def recive_massage_from_sqs(url='https://sqs.eu-central-1.amazonaws.com/332424004628/NameToSerch'):
    # Create SQS client
    sqs = boto3.client('sqs', region_name="eu-central-1", aws_access_key_id='AKIAILK5L3NMS5TLTKOQ',
                       aws_secret_access_key='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')

    # Receive message from SQS queue
    response = sqs.receive_message(
        QueueUrl=url,
        AttributeNames=[
            'SentTimestamp'
        ],
        MaxNumberOfMessages=1,
        MessageAttributeNames=[
            'All'
        ],
        VisibilityTimeout=0,
        WaitTimeSeconds=0
    )

    message = response['Messages'][0]
    receipt_handle = message['ReceiptHandle']

    # Delete received message from queue
    sqs.delete_message(
        QueueUrl=url,
        ReceiptHandle=receipt_handle
    )
    return message["Body"]


def upload_file(file_name, bucket_name, object_name=None):
    if object_name is None:
        object_name = file_name

    # Upload the file
    s3_client = boto3.client('s3', region_name='eu-central-1', aws_access_key_id='AKIAILK5L3NMS5TLTKOQ',
                             aws_secret_access_key='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')
    try:
        response = s3_client.upload_file(file_name, bucket_name, object_name)
    except ClientError as e:
        logging.error(e)
        return False
    return True


def read_image_from_s3(bucket_name, key, region_name='eu-central-1'):
    s3 = boto3.resource('s3', region_name='eu-central-1', aws_access_key_id='AKIAILK5L3NMS5TLTKOQ',
                        aws_secret_access_key='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')
    bucket = s3.Bucket(bucket_name)
    object = bucket.Object(key)
    response = object.get()
    file_stream = response['Body']
    im = Image.open(file_stream)
    return np.array(im)


def download_weigths_from_s3(bucket_name, key, region_name='eu-central-1'):
    s3 = boto3.resource('s3', region_name='eu-central-1', aws_access_key_id='AKIAILK5L3NMS5TLTKOQ',
                        aws_secret_access_key='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')
    bucket = s3.Bucket(bucket_name)
    object = bucket.Object(key)
    response = object.get()
    weigths = response['Body']
    return weigths


"""
master&slave
"""
def master_server():
    HOST = '172.31.37.155'  # private ip
    PORT = 65432  # Port to listen on (non-privileged ports are > 1023)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        conn, addr = s.accept()

        with conn:
            while True:
                data = conn.recv(1024)
                if data:
                    itemname = data.decode(encoding='UTF-8')
                    break

                conn, addr = s.accept()

    return itemname


def start_test(name, weigths_name, children):
    # connect to test bucket
    s3 = boto3.resource('s3', region_name='eu-central-1', aws_access_key_id='AKIAILK5L3NMS5TLTKOQ',
                        aws_secret_access_key='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')
    test_bucket = s3.Bucket("testimages1")

    # transfer all file names from bucket to list
    test_list_of_files = []
    for o in test_bucket.objects.all():
        test_list_of_files.append(o.key)

    length = int(len(test_list_of_files) / num_of_slaves)
    # for every slave that already done train send test with name from user

    for i in range(1, num_of_slaves + 1):

        # send weights name
        comm.send(obj=weigths_name, dest=i, tag=1)
        # send name of person for searching images of
        comm.send(obj=name, dest=i, tag=1)
        # send part of images stock
        if i != num_of_slaves:
            comm.send(obj=test_list_of_files[(i - 1) * length: i * length], dest=i, tag=1)
        else:
            comm.send(obj=test_list_of_files[(i - 1) * length:], dest=i, tag=1)
        children.append(i)
    return children


class face_loader_test(Dataset):

    def __init__(self, path, name_bucket, resize=224, transforms=None, preprocessing=None):
        self.path = path
        # resize the image must be the same size
        self.resize = resize
        self.transforms = transforms
        self.preprocessing = preprocessing
        self.name_bucket = name_bucket

    def __len__(self):
        return len(self.path)

    def __getitem__(self, idx):
        if idx not in range(0, len(self.path)):
            return self.__getitem__(np.random.randint(0, self.__len__()))
        p = self.path[idx]
        image1 = read_image_from_s3(self.name_bucket, p)
        # All image in same size
        image1 = cv2.resize(image1, (224, 224))

        if self.transforms:
            image1 = self.transforms(image=image1)['image']
        image1 = np.transpose(image1, (2, 1, 0))
        image1 = torch.tensor(image1)
        return image1, p


def search(model, name_search, list_test_dict, test_loader, bucket_name):
    model.eval()
    # connect to test bucket
    for index, (img, p) in enumerate(test_loader):

        img = img.float()

        with torch.no_grad():
            out = model(img)

        pred = out.argmax(dim=1, keepdim=True)
        res = pred.cpu().numpy()
        if list_test_dict[res[0][0]] == name_search:
            img1 = img.cpu().data.numpy()
            img1 = img1[0]
            img1 = np.transpose(img1, (2, 1, 0))
            img1 = img1.astype('uint8')
            cv2.imwrite(p[0], img1)

            m = nn.Softmax(dim=1)
            out = m(out)
            out = out.cpu().numpy()

            string_to_send = name_search + "!" + p[0] + "!" + str(out[0][res[0][0]])
            send_sqs_message(string_to_send, "https://sqs.eu-central-1.amazonaws.com/332424004628/ImageAccuracy")
            upload_file(p[0], bucket_name, p[0])


def search_images(name_to_search, files_to_test, bucket_name, weigths):
    # people in system
    list_test_dict = {0: "harel", 1: "anna", 2: 'erdogan'}
    # number of people in the system
    num_class = len(list_test_dict)
    # load model
    resnet18 = models.resnet18(pretrained=False)
    resnet18.fc = nn.Linear(in_features=512, out_features=num_class)
    # load weights to model
    path='/home/ubuntu/'+weigths

    resnet18.load_state_dict(torch.load(path))
    # move model to CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    resnet18 = resnet18.to(device)
    # image loader
    face_test = face_loader_test(files_to_test, 'testimages1', transforms=None)
    test_loader = DataLoader(face_test, batch_size=1, shuffle=True)
    # running test om images
    search(resnet18, name_to_search, list_test_dict, test_loader, bucket_name)


def slave_recive():
    weigths_name = comm.recv(source=0, tag=1)
    # receive name to find
    name_to_search = comm.recv(source=0, tag=1)
    # receive list of file names for testing
    files_to_test = comm.recv(source=0, tag=1)
    return weigths_name, name_to_search, files_to_test


def termination_detection(children):
    while children:
        msg = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG)
        children.remove(msg)
    print("All children of process ", rank, " terminated")



def dfs(graph, node):
    global visited, sequence
    if node not in visited:
        visited.append(node)
        for n in graph[node]:
            sequence.append(n)
            dfs(graph,n)
    return sequence,  visited

def wave(graph, node, token):
    msg =""
    sequence = []
    sequence.append(node)
    s, visited= dfs(graph, node)
    sequence += s

    print("s",sequence)
    time.sleep(rank)

    if rank == sequence[0]:
            comm.send(obj=token, dest=sequence[1], tag=1)
    for i in range(1, len(sequence)-1):
            if rank == sequence[i]:

                msg = comm.recv(source=sequence[i-1], tag=MPI.ANY_TAG)
                comm.send(obj=msg, dest=sequence[i+1], tag=1)
    if rank == sequence[len(sequence)-1]:
            msg = comm.recv(source=sequence[len(sequence)-2], tag=MPI.ANY_TAG)

    sequence.reverse()

    if rank == sequence[0]:
          comm.send(obj=msg, dest=sequence[1], tag=1)
    for i in range(1, len(sequence)-1):
          if rank == sequence[i]:
              msg = comm.recv(source=sequence[i-1], tag=MPI.ANY_TAG)
              comm.send(obj=msg, dest=sequence[i+1], tag=1)
    if rank == sequence[len(sequence)-1]:
          msg = comm.recv(source=sequence[len(sequence)-2], tag=MPI.ANY_TAG)
          print("End wave from rank= ",rank," the message is ",msg)




if rank == 0:
    # wait until waits in bucket and then get weights file name
    weigths_name = master_server()
    # receive name of person to search images of
    name = recive_massage_from_sqs('https://sqs.eu-central-1.amazonaws.com/332424004628/NameToSerch')
    # start testing images
    children = start_test(name, weigths_name, children)
    # check if all children are done
    termination_detection(children)
    # exit program with massage

else:
    # receive data from master
    weigths_name, name_to_search, files_to_test = slave_recive()
    # read weights from s3 bucket
    weigths = download_weigths_from_s3("waigths", weigths_name)
    # start search images and upload them to bucket
    search_images(name_to_search, files_to_test, "imagesofpersonfound", weigths_name)
    # check if all children are done
    termination_detection(children)
    # me and all my children are done
    comm.send(obj=rank, dest=0, tag=1)