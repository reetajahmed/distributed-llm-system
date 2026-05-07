from client.client import run_client
import config

def run_test(scheduler):
    print("\n===== Running Load Test =====\n")
    num_users = config.NUM_USERS
    print(f"Running test with {num_users} users\n")
    run_client(scheduler,num_users)
    print("\n===== Test finished =====\n")
