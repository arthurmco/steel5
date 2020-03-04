main:
    addi x2, x0, 2
    addi x2, x2, 2
    jal x0, 0x8
    nop
    nop
    bne x1, x2, test
    nop
    nop

test:
    addi x1, x0, 1000
    addi x3, x0, 0x200
    sw x2, 0(x3)
    sw x1 4(x3)