import { Component, Input, Output, EventEmitter } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Student } from '../student-input/student-input.component';
import { StudentItemComponent } from '../student-item/student-item.component';

@Component({
  selector: 'app-student-list',
  standalone: true,
  imports: [CommonModule, StudentItemComponent], // CommonModule provides *ngFor, *ngIf
  templateUrl: './student-list.component.html',
  styleUrl: './student-list.component.css'
})
export class StudentListComponent {
  // Receives the array of students from the parent
  @Input() students: Student[] = [];

  // EventEmitter to pass delete events up to the parent
  @Output() studentDeleted = new EventEmitter<number>();

  /**
   * Handles delete event from StudentItemComponent.
   * Passes the student id up to the parent (AppComponent).
   */
  onDeleteStudent(id: number): void {
    this.studentDeleted.emit(id);
  }
}
