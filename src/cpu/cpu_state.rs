use crate::memaccess::Memory;

use crate::opcodes::*;
use crate::instruction::Instruction;

/*
 * Represents a CPU state
 *
 * This includes the registers (x0 to x31) and the program counter
 * Reads on x0 always returns 0, writes have no effect. The other registers are general
 * purpose registers
 *
 * TODO: add floating point registers
 */
#[derive(Debug)]
pub struct CPUState {
    pub x: [u32; 32],
    pc: u32,
}

impl CPUState {
    pub fn new(pc: u32) -> CPUState {
        return CPUState { x: [0; 32], pc };
    }

    /**
     * A wrapper to change register
     *
     * Used to hide the fact that we have to treat 0 specially
     * Reads can be done normally, though
     */
    fn change_register(&mut self, num: u32, val: u32) {
        if num > 0 {
            self.x[num as usize] = val;
        }
    }

    fn execute_register_arithmetic_instruction(
        &mut self,
        fun: OpcodeRegFunctions,
        t: InstructionType,
    ) {
        if let InstructionType::RInst {
            rd,
            funct3,
            rs1,
            rs2,
            funct7,
        } = t
        {
            let r1 = rs1 as usize;
            let r2 = rs2 as usize;
            match fun {
                OpcodeRegFunctions::Value => panic!("what?"),
                OpcodeRegFunctions::ADD_SUB => {
                    if funct7 == 0x100000 {
                        self.change_register(rd, self.x[r2] - self.x[r1]);
                    } else {
                        self.change_register(rd, self.x[r1] + self.x[r2]);
                    }
                }
                OpcodeRegFunctions::SLTU => {
                    self.change_register(rd, if self.x[r1] < self.x[r2] { 1 } else { 0 })
                }
                OpcodeRegFunctions::SLT => {
                    let sr1 = self.x[r1] as i32;
                    let sr2 = self.x[r2] as i32;

                    self.change_register(rd, if sr1 < sr2 { 1 } else { 0 })
                }
                OpcodeRegFunctions::AND => self.change_register(rd, self.x[r1] & self.x[r2]),
                OpcodeRegFunctions::OR => self.change_register(rd, self.x[r1] | self.x[r2]),
                OpcodeRegFunctions::XOR => self.change_register(rd, self.x[r1] ^ self.x[r2]),
                OpcodeRegFunctions::SLL => {
                    self.change_register(rd, self.x[r1] << (self.x[r2] & 0x1f))
                }
                OpcodeRegFunctions::SRL_SRA => {
                    if funct7 == 0x100000 {
                        self.change_register(rd, self.x[r1] >> (self.x[r2] & 0x1f));
                    } else {
                        let sr1 = self.x[r1] as i32;
                        self.change_register(rd, (sr1 >> (self.x[r2] & 0x1f)) as u32);
                    }
                }
            }

            self.pc += 4;
        }
    }

    fn execute_immediate_arithmetic_instruction(
        &mut self,
        fun: OpcodeImmFunctions,
        t: InstructionType,
    ) {
        if let InstructionType::IInst {
            rd,
            funct3,
            rs1,
            imm,
        } = t
        {
            let r1 = rs1 as usize;

            let ir1 = self.x[r1] as i32;
            let uimm = imm as u32;

            match fun {
                OpcodeImmFunctions::Value => panic!("what?"),
                OpcodeImmFunctions::ADDI => self.change_register(rd, (ir1 + imm) as u32),
                OpcodeImmFunctions::SLTI => self.change_register(rd, if ir1 < imm { 1 } else { 0 }),
                OpcodeImmFunctions::SLTIU => {
                    self.change_register(rd, if self.x[r1] < uimm { 1 } else { 0 })
                }
                OpcodeImmFunctions::ANDI => self.change_register(rd, self.x[r1] & uimm),
                OpcodeImmFunctions::ORI => self.change_register(rd, self.x[r1] | uimm),
                OpcodeImmFunctions::XORI => self.change_register(rd, self.x[r1] ^ uimm),
                OpcodeImmFunctions::SLLI => self.change_register(rd, self.x[r1] << (uimm & 0x1f)),
                OpcodeImmFunctions::SRLI_SRAI => {
                    if (uimm >> 5) == 0x100000 {
                        self.change_register(rd, self.x[r1] >> (uimm & 0x1f));
                    } else {
                        self.change_register(rd, (ir1 >> (uimm & 0x1f)) as u32);
                    }
                }
            }

            self.pc += 4;
        }
    }

    fn execute_branch_instructions(&mut self, fun: BranchFunctions, t: InstructionType) {
        if let InstructionType::BInst {
            funct3,
            rs1,
            rs2,
            imm,
        } = t
        {
            let r1 = rs1 as usize;
            let r2 = rs2 as usize;
            let spc = self.pc as i32;

            self.pc = match fun {
                BranchFunctions::Value => panic!("what?"),
                BranchFunctions::BEQ => {
                    if self.x[r1] == self.x[r2] {
                        (spc + imm) as u32
                    } else {
                        self.pc + 4
                    }
                }
                BranchFunctions::BNE => {
                    if self.x[r1] != self.x[r2] {
                        (spc + imm) as u32
                    } else {
                        self.pc + 4
                    }
                }
                BranchFunctions::BLT => {
                    let sr1 = self.x[r1] as i32;
                    let sr2 = self.x[r2] as i32;

                    if sr1 < sr2 {
                        (spc + imm) as u32
                    } else {
                        self.pc + 4
                    }
                }
                BranchFunctions::BLTU => {
                    if self.x[r1] < self.x[r2] {
                        (spc + imm) as u32
                    } else {
                        self.pc + 4
                    }
                }
                BranchFunctions::BGE => {
                    let sr1 = self.x[r1] as i32;
                    let sr2 = self.x[r2] as i32;

                    if sr1 >= sr2 {
                        (spc + imm) as u32
                    } else {
                        self.pc + 4
                    }
                }
                BranchFunctions::BGEU => {
                    if self.x[r1] >= self.x[r2] {
                        (spc + imm) as u32
                    } else {
                        self.pc + 4
                    }
                }
            }
        }
    }

    /**
     * Execute an instruction
     *
     * val is the instruction machine code
     * memory is for managing memory from the load and store instructions
     */
    fn execute_instruction(&mut self, val: u32, mem: &mut Memory) {
        let instruction = Instruction::new(val);

        println!(
            "current instruction at pc {:x}\n\t{:?}",
            self.pc, instruction
        );

        match instruction.opcode {
            Opcode::OP(v) => self.execute_register_arithmetic_instruction(v, instruction.data),
            Opcode::OP_IMM(v) => self.execute_immediate_arithmetic_instruction(v, instruction.data),
            Opcode::BRANCH(v) => self.execute_branch_instructions(v, instruction.data),
            Opcode::FREE_OPS(v) => match v {
                FreeOpcodes::JALR => {
                    if let InstructionType::IInst {
                        rd,
                        funct3: _,
                        rs1,
                        imm,
                    } = instruction.data
                    {
                        let off = self.x[rs1 as usize] as i32;
                        self.change_register(rd, self.pc + 4);

                        self.pc = ((off + imm) & -1) as u32;
                    }
                }
                FreeOpcodes::JAL => {
                    if let InstructionType::JInst { rd, imm } = instruction.data {
                        self.change_register(rd, self.pc + 4);

                        self.pc = (imm & 0xfffffffe) as u32;
                    }
                }
                FreeOpcodes::LUI => {
                    if let InstructionType::UInst { rd, imm } = instruction.data {
                        let val = (imm as u32) << 12;
                        self.change_register(rd, val);
                        self.pc += 4;
                    }
                }
                FreeOpcodes::AUIPC => {
                    if let InstructionType::UInst { rd, imm } = instruction.data {
                        let val = (imm as u32) << 12;
                        self.change_register(rd, val + self.pc);
                        self.pc += 4;
                    }
                }
                FreeOpcodes::LOAD => {
                    if let InstructionType::IInst {
                        rd,
                        funct3,
                        rs1,
                        imm,
                    } = instruction.data
                    {
                        let addr = ((self.x[rs1 as usize] as i32) + imm) as u32;

                        let signed = (val & 0x80000000) == 0x80000000;

                        let sval = match funct3 & 0xf {
                            0 => mem.read8(addr) as i32,
                            1 => mem.read16(addr) as i32,
                            2 => mem.read32(addr) as i32,
                            _ => panic!("Unknown read size {} for LOAD", funct3),
                        };

                        let uval = match funct3 & 0xf {
                            0 => mem.read8(addr) as u32,
                            1 => mem.read16(addr) as u32,
                            2 => mem.read32(addr) as u32,
                            _ => panic!("Unknown read size {} for LOAD", funct3),
                        };

                        self.change_register(rd, if signed { sval as u32 } else { uval })
                    }
                }
                FreeOpcodes::STORE => {
                    if let InstructionType::SInst {
                        rd,
                        funct3,
                        rs1,
                        rs2,
                        imm,
                    } = instruction.data
                    {
                        let addr = ((self.x[rs1 as usize] as i32) + imm) as u32;

                        match funct3 {
                            0 => mem.write8(addr, (rs2 & 0xff) as u8),
                            1 => mem.write16(addr, (rs2 & 0xffff) as u16),
                            2 => mem.write32(addr, rs2),
                            _ => panic!("Unknown write size {} for LOAD", funct3),
                        };

                        self.change_register(rd, val)
                    }
                }
            },
        }
    }

    pub fn run(&mut self, memory: &mut Memory) {
        self.execute_instruction(memory.read32(self.pc), memory)
    }
}
