

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
    x: [u32; 32],
    pc: u32
 }


 /*
  * Instruction types
  *
  * We have 4 types: R, I, S and U, and 2 additional types: B and J
  * opcode will always have 7 bits, funct3 has 3 bits, rd, rs1 and rs2 will
  * have 5 bits, funct7 has 6 bits, immediate varies
  */
#[derive(Debug)]
pub enum InstructionType {
    RInst{rd: u16, funct3: u16, rs1: u16, rs2: u16, funct7: u16},
    IInst{rd: u16, funct3: u16, rs1: u16, imm: u32},
    SInst{rd: u16, funct3: u16, rs1: u16, rs2: u16, imm: u32},

    // B-type has the same fields as S-tyle, but the immediate
    // is spread in memory like this:
    // | imm[12] | imm[10:5] | rs2 | rs1 | funct3 | imm[4:1] | imm[11] | opcode
    BInst{rd: u16, funct3: u16, rs1: u16, rs2: u16, imm: u32},

    // Watch out that we only save the top 20 bits of the immediate
    // in the U-instruction
    UInst{rd: u16, imm: u32},

    // J-type has the same fields as I type, but the immediate is
    // spread like this, in memory
    // | imm[20] | imm[10:1] | imm[11] | imm[19:12] | rd | opcode |
    JInst{rd: u16, imm: u32}
}

#[derive(Debug)]
pub enum OpcodeImmFunctions {
    // Adds imm to value in register rs1
    ADDI,

    // Puts 1 in register rd if value in rs1 is less than the immediate
    // else puts 0
    SLTI,

    // Same as SLTI, but comparison is unsigned
    SLTIU,

    // rd = rs1 AND imm
    ANDI,

    // rd = rs1 OR imm
    ORI,

    // rd = rs1 XOR imm
    XORI,

    // Shifts rs1 by the amount on the lower 5 bits of imm
    // SLLI = shift left, SRLI, shift right,
    // SRAI = arithmetical right shift (preserves sign bit)
    SLLI,
    SRLI,
    SRAI,
}

#[derive(Debug)]
pub enum OpcodeRegFunctions {

    // Adds rs1 to rs2, puts result into rd
    ADD,

    // Same as SLTI, but compares rs1 with rs2 instead of rs1 and imm
    SLT,
    SLTU,

    // rd = rs1 AND rs2
    AND,

    // rd = rs1 OR rs2
    OR,

    // rd = rs1 XOR rs2
    XOR,

    // SLL = same as SLLI, but shifts rs1 by lower 5 bits of rs2
    SLL,

    // Same as SRLI, but shifts right by lower 5 bits of rs2
    SRL,

    // Subtracts rs2 from rs1
    SUB,

    // Same as SRAI but... you understood.
    SRA,

}


/*
 * The opcodes. 
 *
 * Sometimes the opcode will have some enum inside it, the enum values are the values of funct3 for
 * that specific opcode.
 *
 * The RISCV has a NOP instruction, who is basically
 * "addi r0, r0, 0"
 */
#[derive(Debug)]
pub enum Opcode {
    /**** R-type ****/
    OP(OpcodeRegFunctions),

    /**** I-type ****/
    OP_IMM(OpcodeImmFunctions),

    // Jumps to the address specified by rs1 + imm, imm being
    // signed. The address least significant bit is set to
    // zero before jumping to it. rd becomes the address
    // of the instruction following the jump.
    // This generates a instruction address misaligned 
    // exception of the instruction is not aligned to a
    // 4-byte boundary
    JALR,

    /**** S-type ****/
    
    // Stores the value in rs2 into memory address (rs1 + offset)
    STORE,


    /**** B-type ****/
    // All branch instructions, when branching, will get the
    // 12-bit offset from the immediate, sign extend it, add
    // to pc and jump to it.

    // Branch if rs1 and rs2 are equal.
    BEQ,

    // Branch if rs1 and rs2 are different
    BNE,

    // Branch if rs1 < rs2, signed compare
    BLT,

    // Branch if rs1 < rs2, unsigned compare
    BLTU,

    // Branch if rs1 >= rs2, signed cmp
    BGE,

    // Branch if rs1 >= rs2, unsigned cmp
    BGEU,

    /**** U-type ****/

    // Loads the immediate value in the top 20 bits of rd,
    // and fills the lower 12 bits with zeros
    // LUI means "load upper immediate"
    LUI,

    // Forms a 32 bit offset from the 20-bit immediate,
    // fills the lower 12 bits with zeroes, adds to PC
    // and puts the result in rd.
    // AUIPC means "add upper immediate to PC"
    AUIPC,


    /**** J-type ****/
    // Jumps to the offset specified by the immediate
    // (in reality, jumps to pc + offset). The offset is sign
    // extended. Stores the address of the following instruction
    // (pc+4) into rd. Usually rd is x1.
    // Unconditional jumps (j <offset>) are encoded as "jal x0, <offset>"
    JAL,

    // Loads a value from memory address (rs1 + offset) to rd
    LOAD,

}

 /*
  * A RISC-V instruction
  *
  * This includes the opcode plus some additional registers that it uses,
  * or some memory offset, depending on the opcode.
  * We also have fields in the instruction for signaling a future variable-length
  * encoding.
  *
  * All instructions must be aligned on a 32-bit boundary. If the instruction is
  * variable-length, it length must be a multiple of 16-bit.
  *
  * Because of this scheme, all 32-bit RISC-V instructions have the 2-lower order bits
  * set.
  *
  * Also, instructions with the lower 16-bits all zeroed are invalid
  */
  #[derive(Debug)]
  pub struct Instruction {
     opcode: u32,
     data: InstructionType
  }
  
  impl Instruction {
      fn new(inst: u32) -> Instruction {
          return Instruction{opcode: 0, data: InstructionType::UInst{rd:0, imm:0}}
      }
  }


fn main() {
    println!("Hello, world!");
}
