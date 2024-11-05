import re, os
import pandas as pd

# Function to parse a single log entry
def parse_log_entry(log_entry):
    data = {}
    # Regular expressions for matching each line
    patterns = {
        # 'timestamp': r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}',
        # 'selected_clients': r'Selected clients: \[(.*?)\]',
        # 'broadcast_clients': r'Server broadcast to \[(.*?)\] succeed',
        # 'listening_clients': r'Server listening to \[(.*?)\] succeed',
        'global_acc': r'Global acc: (\d+\.\d+)',
        'global_loss': r'Global loss: (\d+\.\d+)',
        'acc_bf_train': r'- cluster_acc: (\d+\.\d+)',
        'loss_bf_train': r'- cluster_loss: (\d+\.\d+)',
        'eval_bf_acc:': r'eval cluster_acc: (\d+\.\d+)',
        'eval_bf_loss:': r'eval cluster_loss: (\d+\.\d+)',
        'acc_bf_train_opt': r'bf_test_optim_acc: (\d+\.\d+)',
        'loss_bf_train_opt': r'bf_test_optim_loss: (\d+\.\d+)',
        'acc_after_train': r'af_test_acc: (\d+\.\d+)',
        'loss_after_train': r'af_test_loss: (\d+\.\d+)',
        'acc_after_train_opt': r'test_acc: (\d+\.\d+)',
        'loss_after_train_opt': r'test_loss: (\d+\.\d+)',
        'train_samples': r'train_samples: (\d+\.\d+)',
        'train_loss': r'train_loss: (\d+\.\d+)',
        'round_time_cost': r'Simulated round time cost: (\d+\.\d+)',
        'slowest_client': r'slowest client (\d+)',
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, log_entry)
        if match:
            # data[key] = match.group(1) if len(match.groups()) == 1 else match.groups()
            data[key] = match.group(1)
    
    return data

# Function to process the log file and extract the data into a DataFrame
def process_log_file(file_path):
    with open(file_path, 'r') as file:
        logs = file.read()
        
    # Split logs by rounds
    rounds = logs.split('==========End of Round')[0:-1]  # Remove the last empty split
    
    data = []
    for round_log in rounds:
        round_data = parse_log_entry(round_log)
        data.append(round_data)
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    return df

# Main function to execute the script
def main():
    store_path = './csv'
    if not os.path.exists(store_path):
        os.makedirs(store_path, exist_ok=True)
    rst_path = 'results/'
    dirs = os.listdir(rst_path)
    for dir in dirs:
        # file_path = f'{rst}/{file_name}/server.log'
        file_path = os.path.join(rst_path, dir, 'server.log')
        if os.path.exists(file_path):
            output_csv = f'{store_path}/{dir}.csv'
            df = process_log_file(file_path)
            df.to_csv(output_csv, index=False)
            print(f"Results saved to {output_csv}")
        else:
            print(f"File {file_path} does not exist")
            continue


if __name__ == '__main__':
    main()
