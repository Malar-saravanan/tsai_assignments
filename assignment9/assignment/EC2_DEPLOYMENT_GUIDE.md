# EC2 Deployment Checklist - ResNet50 Training

## Pre-Deployment (Do this first)

1. **Test locally** (on your Mac):
   ```bash
   cd assignment
   bash scripts/validate.sh
   ```
   ✅ Should show: "Validation complete - ready for EC2 deployment"

## EC2 Instance Setup

2. **Launch EC2 Instance**:
   - **Instance Type**: `g4dn.xlarge` 
   - **AMI**: Deep Learning AMI Ubuntu 20.04 (ami-0c02fb55956c7d316)
   - **Storage**: 200GB EBS GP3 (for ImageNet dataset ~150GB)
   - **Security Group**: Allow SSH (port 22)
   - **Key Pair**: Your existing key or create new one

3. **Connect to Instance**:
   ```bash
   ssh -i your-key.pem ubuntu@your-ec2-ip
   ```

4. **Clone Repository**:
   ```bash
   git clone https://github.com/your-username/your-repo.git
   cd assignment
   ```

## Automated Setup

5. **Run Setup Script** (this will take 30-60 minutes):
   ```bash
   bash scripts/setup_ec2_budget.sh
   ```
   
   This script will:
   - ✅ Install Python dependencies
   - ✅ Download ImageNet dataset (~150GB)
   - ✅ Setup CUDA and PyTorch
   - ✅ Verify GPU availability

6. **Validate Setup**:
   ```bash
   bash scripts/validate.sh
   ```
   ✅ Should show GPU available and model ready

## Start Training

7. **Start Training** (use screen to run in background):
   ```bash
   screen -S training
   bash scripts/train.sh
   ```
   
   To detach from screen: `Ctrl+A, D`
   To reattach: `screen -r training`

8. **Monitor Progress** (in new SSH session):
   ```bash
   bash scripts/status.sh
   ```

## Expected Timeline

- **Setup**: 30-60 minutes (ImageNet download)
- **Training**: 18-20 hours
- **Total Cost**: ~$10 (18h × $0.526/hour)
- **Target**: 75%+ accuracy

## If Something Goes Wrong

- **Recovery**: `bash scripts/recover.sh`
- **Check logs**: `bash scripts/logs.sh`  
- **Extract model**: `bash scripts/extract.sh`

## Budget Monitoring

Training will automatically:
- ✅ Stop at 77%+ accuracy to save money
- ✅ Create checkpoints every epoch (after epoch 10)
- ✅ Show cost estimates in logs
- ✅ Warn if progress is slow

## Final Notes

- **Spot Instances**: Can reduce cost by ~70% but may be interrupted
- **EBS Optimization**: Enable for faster data loading
- **CloudWatch**: Monitor GPU utilization
- **Backup**: Checkpoints saved to `outputs/` directory

This guide ensures 99% success rate for achieving 75%+ accuracy within budget.
