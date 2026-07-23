import torch
import triton
import triton.language as tl

@triton.jit
def _clip_ce_forward(
    logits_ptr, logits_row_stride,
    loss_ptr, logsumexp_ptr, labels_ptr,
    NUM_CLASSES: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    
    logits_ptr += row_idx * logits_row_stride
    loss_ptr += row_idx
    logsumexp_ptr += row_idx
    labels_ptr += row_idx

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < NUM_CLASSES

    label_idx = tl.load(labels_ptr).to(tl.int32)
    logits = tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf")).to(tl.float32)

    c = tl.max(logits, 0)
    logsumexp = c + tl.log(tl.sum(tl.exp(logits - c), 0))

    if label_idx != -100:
        x = tl.load(logits_ptr + label_idx).to(tl.float32)
        loss = logsumexp - x
    else:
        loss = 0.0
        
    tl.store(logsumexp_ptr, logsumexp)
    tl.store(loss_ptr, loss)

@triton.jit
def _clip_ce_backward(
    logits_ptr, logits_row_stride,
    dloss_ptr, dloss_row_stride,
    logsumexp_ptr, labels_ptr,
    NUM_CLASSES: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    block_idx = tl.program_id(1)

    logits_ptr += row_idx * logits_row_stride
    dloss_ptr += row_idx * dloss_row_stride
    col_offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < NUM_CLASSES
    
    label_idx = tl.load(labels_ptr + row_idx).to(tl.int32)

    if label_idx != -100:
        dloss = tl.load(dloss_ptr)
    else:
        dloss = 0.0

    x = tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf")).to(tl.float32)
    logsumexp = tl.load(logsumexp_ptr + row_idx)
    
    y = tl.exp(x - logsumexp)
    y = tl.where(col_offsets == label_idx, y - 1.0, y)

    tl.store(logits_ptr + col_offsets, dloss * y, mask=mask)

class FastCLIPCrossEntropyLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, labels):
        batch_size, num_classes = logits.shape
        device = logits.device

        BLOCK_SIZE = triton.next_power_of_2(num_classes)
        if BLOCK_SIZE < 16:
            BLOCK_SIZE = 16
            
        num_warps = 4
        if BLOCK_SIZE >= 2048: num_warps = 8
        if BLOCK_SIZE >= 8192: num_warps = 16

        losses = torch.empty(batch_size, dtype=torch.float32, device=device)
        logsumexp = torch.empty(batch_size, dtype=torch.float32, device=device)

        _clip_ce_forward[(batch_size,)](
            logits, logits.stride(0),
            losses, logsumexp, labels,
            NUM_CLASSES=num_classes,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )

        ctx.save_for_backward(logits, logsumexp, labels)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        return losses

    @staticmethod
    def backward(ctx, dlosses):
        logits, logsumexp, labels = ctx.saved_tensors
        batch_size, num_classes = logits.shape
        
        _clip_ce_backward[(batch_size, 1)](
            logits, logits.stride(0),
            dlosses, dlosses.stride(0),
            logsumexp, labels,
            NUM_CLASSES=num_classes,
            BLOCK_SIZE=ctx.BLOCK_SIZE,
            num_warps=4,
        )
        return logits, None

def fast_clip_cross_entropy_loss(logits, labels):
    loss = FastCLIPCrossEntropyLoss.apply(logits, labels)
    n_items = torch.count_nonzero(labels != -100)
    return loss.sum() / n_items