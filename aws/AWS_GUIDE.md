# Running the pipeline on AWS EC2

The laptop builds `upload.tgz` (code + data). The EC2 instance trains on the full training data,
predicts the test set and validates the output. You download the two TSVs. Expected: roughly
1–1.5 h and a few dollars of credit on `c7i.8xlarge`.

In the commands below, replace `<IP>` with the instance's public IPv4 address. `$key` is the path
to your `.pem` key file.

---

## 0. Build the bundle (laptop, PowerShell, in `D:\Amazon_ml`)
```powershell
powershell -ExecutionPolicy Bypass -File aws\pack.ps1      # creates D:\Amazon_ml\upload.tgz
```

## 1. Launch the instance (AWS console → EC2 → Launch instance)
| Setting | Value |
|---|---|
| AMI | **Ubuntu Server 24.04 LTS**, 64-bit (x86) |
| Instance type | **c7i.8xlarge** (32 vCPU, 64 GB). If AWS refuses because of a vCPU limit, use **c7i.4xlarge** (16 vCPU, 32 GB) |
| Key pair | *Create new key pair* → RSA, `.pem` → save as `C:\Users\<you>\.ssh\amazon-ml.pem` |
| Network | *Allow SSH traffic from* → **My IP** |
| Storage | **60 GiB gp3** |

Click **Launch**, open the instance and copy its **Public IPv4 address**.

## 2. Lock down the key file (laptop, PowerShell, once)
Windows OpenSSH refuses keys that other users can read:
```powershell
$key = "$env:USERPROFILE\.ssh\amazon-ml.pem"
icacls $key /inheritance:r
icacls $key /grant:r "$($env:USERNAME):(R)"
```

## 3. Upload the bundle (laptop, PowerShell, in `D:\Amazon_ml`)
```powershell
scp -i $key upload.tgz ubuntu@<IP>:~/
```
Answer `yes` to the first-connection fingerprint question.

## 4. Start the run (on the instance)
```powershell
ssh -i $key ubuntu@<IP>
```
Then on the instance:
```bash
mkdir -p ~/Amazon_ml && tar -xzf ~/upload.tgz -C ~/Amazon_ml
tmux new -s er                          # keeps the job alive if you disconnect
cd ~/Amazon_ml && bash aws/run_on_ec2.sh
```
- Detach with `Ctrl-b`, then `d`. You can close the window and the job keeps running.
- Come back later with `ssh -i $key ubuntu@<IP>` and then `tmux attach -t er`.
- Follow progress with `tail -f ~/Amazon_ml/run.log`.

It is finished when the log ends with `PASS` (from the official validator) and then `DONE`.

## 5. Download the results (laptop, PowerShell, in `D:\Amazon_ml`)
```powershell
mkdir output -Force
scp -i $key "ubuntu@<IP>:~/Amazon_ml/output/*.tsv" output\
scp -i $key -r "ubuntu@<IP>:~/Amazon_ml/code/business_entity_resolution/artifacts" code\business_entity_resolution\
scp -i $key "ubuntu@<IP>:~/Amazon_ml/run.log" output\
```
The second command (the trained model) and the third (the log, for the write-up) are optional.

## 6. Terminate the instance
Console → EC2 → Instances → select it → **Instance state → Terminate**. A *stopped* instance
still bills for its disk; terminate it once you have the files.

## 7. Submit
Upload `output\matching_results.tsv` to the challenge portal.

---

### If something goes wrong
- **The run stopped part-way:** run `bash aws/run_on_ec2.sh` again. The normalised caches are reused.
- **Only prediction needs redoing** (the model is already trained): `bash aws/run_on_ec2.sh predict`.
- **`Permission denied (publickey)`:** check the user is `ubuntu`, the key is the right `.pem`, and step 2 was done.
- **SSH times out:** your IP changed. Edit the instance's security group so the SSH rule uses *My IP* again.
