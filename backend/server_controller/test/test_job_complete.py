import requests

BASE = "https://ky3libvtbr544cocs4lk4lqp6q.apigateway.ap-hyderabad-1.oci.customer-oci.com/imgvidgen"

def test_job_complete(job_id: str):
    payload = {
        "job_id": job_id,
        "result_r2_key": f"quickpod/{job_id}.png",
        "failed": False
    }
    r = requests.post(f"{BASE}/job_complete", json=payload, verify=False)
    print("Status:", r.status_code)
    print("Response:", r.json())

if __name__ == "__main__":
    # Replace with a real job_id from test_submit
    test_job_complete("your-job-id-here")
