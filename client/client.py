from common.types import Request
from concurrent.futures import ThreadPoolExecutor
import time
import config


def send_request(scheduler, request_id):
    query = f"What is AI? Request {request_id}"
    request = Request(id=request_id, query=query)
    print(f"[Client] Sending request {request_id}")
    scheduler.submit_request(request)


def run_client(scheduler, num_requests=None):
    if num_requests is None:
        num_requests = config.NUM_USERS
    
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
        futures = []
        for i in range(num_requests):
            futures.append(executor.submit(send_request, scheduler, i))
            time.sleep(config.REQUEST_DELAY)

        # wait for all requests to complete
        for future in futures:
            future.result()

    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f} seconds")

