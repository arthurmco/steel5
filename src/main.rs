mod memaccess;
use memaccess::Memory;

mod cpu;
use cpu::*;

fn main() {
    println!("Hello, world!");

    // Machine code examples extracted from here
    //  - https://passlab.github.io/CSCE513/notes/lecture04_RISCV_ISA.pdf
    //  - https://inst.eecs.berkeley.edu/~cs61c/fa17/disc/4/Disc4Sol.pdf
    println!("{:?}", Instruction::new(0xFFF28413)); //addi s0 t0 -1
    println!("{:?}", Instruction::new(0x007302b3)); //add x5, x6, x7
    println!("{:?}", Instruction::new(0x00650333)); //add x6, x10, x6
    println!("{:?}", Instruction::new(0x00512a03)); //lw s4 5(sp)
    println!("{:?}", Instruction::new(0x60000113)); //addi sp, 0, 1536
    println!("{:?}", Instruction::new(0x00000013)); //addi x0, x0, 0 aka NOP
    println!("{:?}", Instruction::new(0x0400026f)); //jal x4, #64
    println!("{:?}", Instruction::new(0x0400026f)); //jal x4, #64
    println!("{:?}", Instruction::new(0x02419063)); //bne r3, r4, #32

    let mut cpu = CPUState::new(100);
    let mut memory = Memory::new();

    memory.write8(0, 1);
    memory.write8(1, 2);
    memory.write8(2, 3);
    memory.write8(3, 4);
    memory.write8(4, 0x20);
    memory.write8(5, 0x21);
    memory.write8(6, 0x22);
    memory.write8(7, 0x23);
    memory.write8(8, 0x24);

    println!("{:x?}", cpu);
    println!("{:?}", memory);
    println!("{:x}", memory.read32(0)); // needs to be 0x04030201
    println!("{:x}", memory.read32(4)); // needs to be 0x23222120
    println!("{:x}", memory.read16(0)); // needs to be 0x0201
    println!("{:x}", memory.read16(2)); // needs to be 0x0403
    println!("{:x}", memory.read8(8)); // needs to be 0x24

    memory.write32(64, 0x00512a03);

    memory.write32(100, 0xfff28413);
    memory.write32(104, 0xfff28413);
    memory.write32(108, 0x02419063);
    memory.write32(112, 0x0400026f);
    println!("{:?}", Instruction::new(memory.read32(100))); //addi s0 t0 -1

    cpu.run(&mut memory);
    cpu.run(&mut memory);
    cpu.run(&mut memory);
    cpu.run(&mut memory);
    cpu.run(&mut memory);

    println!("{:x?}", cpu);
    assert_eq!(cpu.x[20], 0x24232221);
}
