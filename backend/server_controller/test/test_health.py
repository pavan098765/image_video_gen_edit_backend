import requests

BASE = "https://ky3libvtbr544cocs4lk4lqp6q.apigateway.ap-hyderabad-1.oci.customer-oci.com/imgvidgen"

def test_health():
    r = requests.get(f"{BASE}/health", verify=False)  # verify=False to skip SSL issues
    print("Status:", r.status_code)
    print("Response:", r.json())

if __name__ == "__main__":
    test_health()
