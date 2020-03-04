# steel5

A RISC-V emulator written in Rust, because I wanted to do something in Rust :smile:

steel5 will implement the 32-bit RISC-V instruction set (RV32I), and the 64-bit one after
that. 

For now, it does not run any instruction.

Check the docs: https://github.com/riscv/riscv-isa-manual/releases/tag/draft-20200229-27b40fb

Some useful links:
 - https://github.com/rv8-io/rv8/blob/master/doc/pdf/riscv-instructions.pdf

## Supported hardware

I am aiming to support the same hardware as the QEMU-port (https://github.com/riscv/riscv-qemu/wiki), if possible

 - HTIF Console (Host Target Interface)
 - SiFive CLINT (Core Local Interruptor) for Timer interrupts and IPIs
 - SiFive PLIC (Platform Level Interrupt Controller)
 - SiFive Test (Test Finisher)
 - SiFive UART, PRCI, AON, PWM, QSPI support 
 - VirtIO MMIO
 - GPEX PCI support
 - Generic 16550A UART emulation
 - MTTCG and SMP support (PLIC and CLINT)

If this becomes hard or impossible to do, this list will change

My emulator will also support a MMU. There is not much to be found, but
I will search for it.

## Objectives

The final objective is to run Linux and NetBSD. I am almost sure  both have risc-v support.

