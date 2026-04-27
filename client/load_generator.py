from client.client import run_client

def run_test(scheduler):
    print("\n===== Running Load Test =====\n")
    num_users=10
    print(f"Running test with {num_users} users\n")
    run_client(scheduler,num_users)
    print("\n===== Test finished =====\n")